"""Untrusted JPEG 2000 must be rejected before Pillow decodes it.

Pillow 12.1.1 wheels bundle OpenJPEG 2.5.4. In that library, opj_j2k_read_sod
writes tp_index[TPsot] with no null or bounds check. Pillow reaches that write
from Image.load(), which these product paths call on attacker-supplied bytes:

- project upload token estimation (Image.open + ImageOps.exif_transpose)
- PDF image extraction when it is enabled (page.images, then Image.save)
- image summarization of files larger than 20 MB (Image.open + thumbnail)

These tests fail while those paths still decode JPEG 2000.
"""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import tempfile
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
_PNG_PDF = (
    Path(__file__).resolve().parent / "fixtures" / "with_image.pdf"
)
_TWENTY_MB = 20 * 1024 * 1024


def _benign_jpeg2000() -> bytes:
    buffer = BytesIO()
    Image.new("L", (8, 8), 128).save(buffer, format="JPEG2000")
    return buffer.getvalue()


def _benign_jpeg() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 8), (10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _tile_part(tile_part_index: int) -> bytes:
    """One SOT + SOD. TNsot is 0 so a valid TLM is not resized to fit TPsot."""
    body = bytes((0, 0, 0, 0, 0, 0x12, tile_part_index & 0xFF, 0))
    return (
        b"\xff\x90"
        + (len(body) + 2).to_bytes(2, "big")
        + body
        + b"\xff\x93"
        + b"\x80\x80\x80\x80"
    )


def _crafted_jpeg2000() -> bytes:
    """Codestream whose TLM allocates one tile-part slot and whose later SOT
    markers carry a larger TPsot. OpenJPEG 2.5.4 writes past that slot."""
    encoded = _benign_jpeg2000()
    soc = encoded.find(b"\xff\x4f")
    codestream = encoded[soc:]
    comment = codestream.find(b"\xff\x64")
    main_header = codestream[:comment]
    tlm_body = bytes((0, 0, 0, 0x12))
    tlm = b"\xff\x55" + (len(tlm_body) + 2).to_bytes(2, "big") + tlm_body
    parts = b"".join(_tile_part(index) for index in range(8))
    return main_header + tlm + parts + b"\xff\xd9"


def _pdf_with_jpx(image: bytes) -> bytes:
    contents = b"q\n8 0 0 8 0 0 cm\n/Im0 Do\nQ\n"
    image_object = (
        b"<< /Type /XObject /Subtype /Image /Width 8 /Height 8 "
        b"/ColorSpace /DeviceGray /BitsPerComponent 8 /Filter /JPXDecode /Length "
        + str(len(image)).encode("ascii")
        + b" >>\nstream\n"
        + image
        + b"\nendstream"
    )
    contents_object = (
        b"<< /Length "
        + str(len(contents)).encode("ascii")
        + b" >>\nstream\n"
        + contents
        + b"endstream"
    )
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 8 8] "
        b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>",
        image_object,
        contents_object,
    ]
    blob = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(blob))
        blob += f"{index} 0 obj\n".encode("ascii") + body + b"\nendobj\n"
    xref_at = len(blob)
    xref = bytearray(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    xref += b"0000000000 65535 f \n"
    for offset in offsets:
        xref += f"{offset:010d} 00000 n \n".encode("ascii")
    trailer = (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode("ascii")
    return bytes(blob + xref + trailer)


def _run_child(mode: str, payload: bytes) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "input.bin"
        path.write_bytes(payload)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(_BACKEND_ROOT), env.get("PYTHONPATH", "")]
        )
        # Abort in the child if the decode overflows the tile-part index.
        env["MALLOC_CHECK_"] = "3"
        env["PYTHONMALLOC"] = "malloc"
        return subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), mode, str(path)],
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )


def _assert_rejected(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
    )
    assert "REJECTED" in result.stdout
    assert "DECODED" not in result.stdout
    assert "EXTRACTED" not in result.stdout


def _worker(mode: str, path: str) -> None:
    data = Path(path).read_bytes()
    if mode == "upload":
        from fastapi import UploadFile
        from PIL import UnidentifiedImageError

        from onyx.server.features.projects.projects_file_utils import (
            estimate_image_tokens_for_upload,
        )

        upload = UploadFile(filename="photo.jpg", file=BytesIO(data), size=len(data))
        try:
            count = estimate_image_tokens_for_upload(upload)
        except (UnidentifiedImageError, OSError):
            gc.collect()
            print("REJECTED")
            return
        gc.collect()
        print(f"DECODED {count}")
        raise SystemExit(1)

    if mode == "upload-allow":
        from fastapi import UploadFile

        from onyx.server.features.projects.projects_file_utils import (
            estimate_image_tokens_for_upload,
        )

        upload = UploadFile(filename="photo.jpg", file=BytesIO(data), size=len(data))
        count = estimate_image_tokens_for_upload(upload)
        gc.collect()
        print(f"TOKENS {count}")
        return

    if mode in {"pdf", "pdf-off"}:
        from onyx.file_processing.extract_file_text import read_pdf_file

        _text, _metadata, images = read_pdf_file(
            BytesIO(data),
            extract_images=mode == "pdf",
        )
        gc.collect()
        if images:
            print(f"EXTRACTED {len(images)}")
            raise SystemExit(1)
        print("REJECTED")
        return

    if mode == "pdf-allow":
        from onyx.file_processing.extract_file_text import read_pdf_file

        _text, _metadata, images = read_pdf_file(BytesIO(data), extract_images=True)
        gc.collect()
        print(f"IMAGES {len(images)}")
        if len(images) != 1 or len(images[0][0]) == 0:
            raise SystemExit(1)
        return

    if mode == "resize":
        from PIL import UnidentifiedImageError

        from onyx.file_processing.image_summarization import (
            UnsupportedImageFormatError,
        )
        from onyx.file_processing.image_summarization import prepare_image_bytes

        try:
            prepare_image_bytes(data)
        except (UnidentifiedImageError, OSError, UnsupportedImageFormatError):
            gc.collect()
            print("REJECTED")
            return
        gc.collect()
        print("DECODED")
        raise SystemExit(1)

    raise SystemExit(f"unknown mode {mode}")


@pytest.mark.parametrize("kind", ["well_formed", "tile_part_overflow"])
def test_project_upload_rejects_jpeg2000(kind: str) -> None:
    payload = _benign_jpeg2000() if kind == "well_formed" else _crafted_jpeg2000()
    _assert_rejected(_run_child("upload", payload))


def test_project_upload_still_estimates_jpeg() -> None:
    result = _run_child("upload-allow", _benign_jpeg())
    assert result.returncode == 0, result.stderr
    assert "TOKENS" in result.stdout
    token_count = int(result.stdout.strip().split()[-1])
    assert token_count > 0


@pytest.mark.parametrize("kind", ["well_formed", "tile_part_overflow"])
def test_enabled_pdf_extraction_rejects_jpeg2000(kind: str) -> None:
    image = _benign_jpeg2000() if kind == "well_formed" else _crafted_jpeg2000()
    _assert_rejected(_run_child("pdf", _pdf_with_jpx(image)))


def test_disabled_pdf_extraction_does_not_decode_jpeg2000() -> None:
    _assert_rejected(_run_child("pdf-off", _pdf_with_jpx(_crafted_jpeg2000())))


def test_enabled_pdf_extraction_still_returns_png() -> None:
    result = _run_child("pdf-allow", _PNG_PDF.read_bytes())
    assert result.returncode == 0, result.stderr
    assert "IMAGES 1" in result.stdout


@pytest.mark.parametrize("kind", ["well_formed", "tile_part_overflow"])
def test_summarization_rejects_jpeg2000_over_20mb(kind: str) -> None:
    image = _benign_jpeg2000() if kind == "well_formed" else _crafted_jpeg2000()
    payload = image + bytes(_TWENTY_MB + 1 - len(image))
    assert len(payload) > _TWENTY_MB
    _assert_rejected(_run_child("resize", payload))


if __name__ == "__main__":
    _worker(sys.argv[1], sys.argv[2])
