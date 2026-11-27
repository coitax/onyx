"""Stored PDF images must match a TIFF written to a real file.

Pillow 12.1.1 writes compressed TIFF data through libtiff into a buffer when the
target has no usable fileno(), which includes io.BytesIO. That writer sizes its
allocation from the previous buffer length and does not clear bytes libtiff
seeks over. For compressed output larger than 64 KiB those gaps can contain
stale heap bytes. Saving the same image to a real file descriptor does not.

read_pdf_file does that BytesIO save for every embedded image when image
extraction is on: it opens the image pypdf produced and saves it again with
format=image.format. A CCITT fax image in a PDF becomes a compressed TIFF, so
the re-save takes the vulnerable path. These tests fail while that Pillow
behavior is present and pass once the stored bytes are identical to a
file-descriptor save (Pillow 13.0.0 zeroes the gaps).
"""

import io
import random
import tempfile
from collections.abc import Callable
from unittest.mock import patch

from PIL import Image

from onyx.file_processing.extract_file_text import extract_text_and_images
from onyx.file_processing.extract_file_text import read_pdf_file

# Group4 output for this page is about 1 MiB, above the 64 KiB gap threshold.
_PAGE_SIDE = 2048
# This page's Group 4 layout leaves a one-byte gap. The byte is stale heap
# and is not zero on every save, so repeat enough times to observe it.
_IMAGE_SEED = 3
_REPETITIONS = 12

ImageExtractor = Callable[[bytes], list[tuple[bytes, str]]]


def _group4_strip(width: int, height: int) -> bytes:
    """One CCITT Group 4 strip, large enough that a TIFF re-save exceeds 64 KiB."""
    raw = random.Random(_IMAGE_SEED).randbytes((width * height) // 8)
    image = Image.frombytes("1", (width, height), raw)
    with tempfile.NamedTemporaryFile(suffix=".tif") as handle:
        image.save(
            handle.name,
            format="TIFF",
            compression="group4",
            strip_size=1 << 30,
        )
        with open(handle.name, "rb") as stored:
            data = stored.read()
        with Image.open(handle.name) as opened:
            offsets = opened.tag_v2[273]
            counts = opened.tag_v2[279]
        if isinstance(offsets, int):
            offsets = (offsets,)
            counts = (counts,)
        return b"".join(data[offset : offset + count] for offset, count in zip(offsets, counts))


def _pdf_with_ccitt_image(stream: bytes, width: int, height: int) -> bytes:
    """Minimal PDF whose only image is a CCITT Group 4 fax stream."""
    content = b"q 100 0 0 100 0 0 cm /Im0 Do Q"
    image = (
        f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
        f"/ColorSpace /DeviceGray /BitsPerComponent 1 /Filter /CCITTFaxDecode "
        f"/DecodeParms << /K -1 /Columns {width} /Rows {height} /BlackIs1 true >> "
        f"/Length {len(stream)} >>\nstream\n"
    ).encode() + stream + b"\nendstream"
    contents = f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
    bodies = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 100 100] "
        b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>",
        image,
        contents,
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(bodies, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer << /Size {len(bodies) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    )
    return bytes(out)


def _file_descriptor_tiff(image_bytes: bytes) -> bytes:
    """Re-encode through a real file descriptor, which Pillow does not leak into."""
    with Image.open(io.BytesIO(image_bytes)) as image:
        image.load()
        compression = image.info.get("compression")
        save_kwargs: dict[str, str] = {}
        if isinstance(compression, str):
            save_kwargs["compression"] = compression
        with tempfile.NamedTemporaryFile(suffix=".tif") as handle:
            image.save(handle.name, format=image.format, **save_kwargs)
            with open(handle.name, "rb") as stored:
                return stored.read()


def _assert_matches_file_save(image_bytes: bytes) -> None:
    assert len(image_bytes) > 64 * 1024
    safe = _file_descriptor_tiff(image_bytes)
    if image_bytes == safe:
        return
    limit = min(len(image_bytes), len(safe))
    mismatch = next(
        (
            index
            for index in range(limit)
            if image_bytes[index] != safe[index]
        ),
        None,
    )
    if mismatch is None:
        detail = f"length {len(image_bytes)} != {len(safe)}"
    else:
        detail = (
            f"offset {mismatch}: stored {image_bytes[mismatch]} "
            f"!= file {safe[mismatch]}"
        )
    raise AssertionError(
        "compressed TIFF saved for a PDF image differs from the same image "
        f"written to a real file ({detail})"
    )


def _ccitt_pdf() -> bytes:
    stream = _group4_strip(_PAGE_SIDE, _PAGE_SIDE)
    assert len(stream) > 64 * 1024
    return _pdf_with_ccitt_image(stream, _PAGE_SIDE, _PAGE_SIDE)


def _repeat(extract: ImageExtractor, pdf: bytes) -> None:
    for _ in range(_REPETITIONS):
        images = extract(pdf)
        assert len(images) == 1
        _assert_matches_file_save(images[0][0])


def test_read_pdf_file_ccitt_resave_matches_file_descriptor() -> None:
    pdf = _ccitt_pdf()

    def extract(payload: bytes) -> list[tuple[bytes, str]]:
        _text, _metadata, images = read_pdf_file(io.BytesIO(payload), extract_images=True)
        return list(images)

    _repeat(extract, pdf)


def test_read_pdf_file_ccitt_callback_matches_file_descriptor() -> None:
    pdf = _ccitt_pdf()

    def extract(payload: bytes) -> list[tuple[bytes, str]]:
        collected: list[tuple[bytes, str]] = []

        def callback(data: bytes, name: str) -> None:
            collected.append((data, name))

        _text, _metadata, images = read_pdf_file(
            io.BytesIO(payload),
            extract_images=True,
            image_callback=callback,
        )
        assert images == []
        return collected

    _repeat(extract, pdf)


def test_extract_text_and_images_ccitt_matches_file_descriptor_when_enabled() -> None:
    """The connector and user-file path stores these bytes when the admin setting is on."""
    pdf = _ccitt_pdf()

    def extract(payload: bytes) -> list[tuple[bytes, str]]:
        with (
            patch(
                "onyx.file_processing.extract_file_text.get_image_extraction_and_analysis_enabled",
                return_value=True,
            ),
            patch(
                "onyx.file_processing.extract_file_text.get_unstructured_api_key",
                return_value=None,
            ),
        ):
            result = extract_text_and_images(io.BytesIO(payload), "fax.pdf")
        return list(result.embedded_images)

    _repeat(extract, pdf)


def test_extract_text_and_images_ccitt_callback_matches_file_descriptor() -> None:
    """SharePoint and Drupal store the callback bytes rather than the returned list."""
    pdf = _ccitt_pdf()

    def extract(payload: bytes) -> list[tuple[bytes, str]]:
        collected: list[tuple[bytes, str]] = []

        def callback(data: bytes, name: str) -> None:
            collected.append((data, name))

        with (
            patch(
                "onyx.file_processing.extract_file_text.get_image_extraction_and_analysis_enabled",
                return_value=True,
            ),
            patch(
                "onyx.file_processing.extract_file_text.get_unstructured_api_key",
                return_value=None,
            ),
        ):
            result = extract_text_and_images(
                io.BytesIO(payload),
                "fax.pdf",
                image_callback=callback,
            )
        assert result.embedded_images == []
        return collected

    _repeat(extract, pdf)
