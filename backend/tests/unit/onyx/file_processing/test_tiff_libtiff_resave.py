"""Re-saving a compressed TIFF must not crash or copy process memory.

Pillow through 12.3.0 passes attacker-controlled TIFF tag values to libtiff's
TIFFSetField with a calling convention libtiff does not use for that tag.
read_pdf_file reaches that encoder: pypdf turns an LZWDecode image XObject
whose decoded bytes are a TIFF into a Pillow image and saves it as TIFF, then
read_pdf_file saves it again with format=image.format. These tests build that
PDF and run the product functions in a child process.

They fail on Pillow 12.1.1 (the child dies with SIGSEGV, or the saved tag
holds bytes that were not in the upload). They pass once the encoder rejects
the mismatched tag instead of reading process memory.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from PIL import Image
from pypdf._codecs._codecs import LzwCodec

_LONG = 4
_BENIGN_TAG = 65000
_DNG_PRIVATE_DATA = 50740
_ACTIVE_AREA = 50829
_COUNT = 4096
_ACTIVE_AREA_VALUE = 64

# 16x16 LZW RGB re-saves to a few hundred bytes. A disclosed tag of `_COUNT`
# bytes makes the output several kilobytes larger.
_MAX_SAFE_IMAGE_BYTES = 2000


def _craft_lzw_tiff(tag: int, value: int) -> bytes:
    """Minimal LZW RGB TIFF plus one inline LONG count-1 entry.

    The extra entry is appended as a new IFD and the header is pointed at it,
    which is how a TIFF writer stores a tag Pillow will load into tag_v2 and
    forward on save. The value fits in the IFD entry, so no extra payload is
    added to the file.
    """
    base = BytesIO()
    Image.new("RGB", (16, 16), (255, 0, 0)).save(
        base, format="TIFF", compression="tiff_lzw"
    )
    data = bytearray(base.getvalue())
    if data[:2] != b"II" or struct.unpack_from("<H", data, 2)[0] != 42:
        raise RuntimeError("Pillow did not write a little-endian classic TIFF")
    ifd_offset = struct.unpack_from("<I", data, 4)[0]
    count = struct.unpack_from("<H", data, ifd_offset)[0]
    entries: list[bytes] = []
    position = ifd_offset + 2
    for _ in range(count):
        entries.append(bytes(data[position : position + 12]))
        position += 12
    next_ifd = struct.unpack_from("<I", data, position)[0]
    entries.append(struct.pack("<HHII", tag, _LONG, 1, value))
    entries.sort(key=lambda entry: struct.unpack_from("<H", entry, 0)[0])
    new_offset = len(data)
    data += struct.pack("<H", len(entries))
    for entry in entries:
        data += entry
    data += struct.pack("<I", next_ifd)
    struct.pack_into("<I", data, 4, new_offset)
    return bytes(data)


def _pdf_with_lzw_image(payload: bytes) -> bytes:
    """PDF whose only image is the payload behind the LZWDecode filter.

    pypdf decodes that filter and, for LZWDecode, opens the result as a TIFF
    and saves it again. The codec here is the one pypdf's decoder uses, so
    the bytes that come back out are the crafted TIFF.
    """
    encoded = LzwCodec().encode(payload)
    if LzwCodec().decode(encoded) != payload:
        raise RuntimeError("LZW codec did not round-trip the TIFF")

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 72 72] "
        b"/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>",
        (
            f"<< /Type /XObject /Subtype /Image /Width 16 /Height 16 "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /LZWDecode "
            f"/Length {len(encoded)} >>\nstream\n".encode()
            + encoded
            + b"\nendstream"
        ),
        b"<< /Length 0 >>\nstream\n\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{index} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)


def _tag_summary(image_bytes: bytes, tag: int, source_tiff: bytes) -> dict[str, Any]:
    with Image.open(BytesIO(image_bytes)) as image:
        image_format = image.format
        value: Any = None
        if hasattr(image, "tag_v2"):
            value = image.tag_v2.get(tag)
    disclosed = False
    if isinstance(value, (bytes, bytearray)) and len(value) >= _COUNT:
        disclosed = bytes(value) not in source_tiff
    if isinstance(value, (bytes, bytearray)):
        shown: Any = {"type": "bytes", "length": len(value)}
    elif isinstance(value, tuple):
        shown = {"type": "tuple", "length": len(value)}
    else:
        shown = {"type": type(value).__name__, "value": value}
    return {
        "format": image_format,
        "size": len(image_bytes),
        "tag": shown,
        "disclosed": disclosed,
    }


def _run_child(entry: str, tag: int, value: int) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[5]
    backend_root = repo_root / "backend"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(backend_root), str(repo_root), env.get("PYTHONPATH", "")]
    )
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), entry, str(tag), str(value)],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    result_line = ""
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            result_line = line[len("RESULT ") :]
    parsed: dict[str, Any] = {}
    if result_line:
        loaded = json.loads(result_line)
        if isinstance(loaded, dict):
            parsed = loaded
    return {
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "result": parsed,
    }


def _assert_safe(completed: dict[str, Any]) -> dict[str, Any]:
    assert completed["returncode"] == 0, (
        f"image re-save exited {completed['returncode']}\n"
        f"stdout:\n{completed['stdout']}\nstderr:\n{completed['stderr']}"
    )
    result = completed["result"]
    assert result, f"child produced no RESULT line:\n{completed['stdout']}"
    images = result["images"]
    assert isinstance(images, list)
    for image in images:
        assert image["disclosed"] is False
        assert image["size"] < _MAX_SAFE_IMAGE_BYTES
    return result


def _child_read_pdf(tag: int, value: int, *, extract_images: bool, callback: bool) -> None:
    from onyx.file_processing.extract_file_text import read_pdf_file

    source = _craft_lzw_tiff(tag, value)
    pdf = BytesIO(_pdf_with_lzw_image(source))
    collected: list[tuple[bytes, str]] = []

    def _callback(data: bytes, name: str) -> None:
        collected.append((data, name))

    _text, _metadata, images = read_pdf_file(
        pdf,
        extract_images=extract_images,
        image_callback=_callback if callback else None,
    )
    produced = list(collected) if callback else list(images)
    summaries = [_tag_summary(data, tag, source) for data, _name in produced]
    print("RESULT " + json.dumps({"images": summaries}), flush=True)


def _child_extract_text_and_images(tag: int, value: int) -> None:
    import onyx.file_processing.extract_file_text as extract_file_text

    def _extraction_enabled() -> bool:
        return True

    def _no_unstructured_key() -> str | None:
        return None

    # The product call site reads these two settings before re-saving. Force
    # the image-extraction branch on, and skip the unstructured API, which
    # would otherwise look up a key store this test does not run.
    extract_file_text.get_image_extraction_and_analysis_enabled = _extraction_enabled  # type: ignore[assignment]
    extract_file_text.get_unstructured_api_key = _no_unstructured_key  # type: ignore[assignment]

    source = _craft_lzw_tiff(tag, value)
    result = extract_file_text.extract_text_and_images(
        BytesIO(_pdf_with_lzw_image(source)),
        "upload.pdf",
    )
    summaries = [
        _tag_summary(data, tag, source) for data, _name in result.embedded_images
    ]
    print("RESULT " + json.dumps({"images": summaries}), flush=True)


def _main() -> None:
    entry = sys.argv[1]
    tag = int(sys.argv[2])
    value = int(sys.argv[3])
    if entry == "read_pdf":
        _child_read_pdf(tag, value, extract_images=True, callback=False)
    elif entry == "read_pdf_callback":
        _child_read_pdf(tag, value, extract_images=True, callback=True)
    elif entry == "read_pdf_off":
        _child_read_pdf(tag, value, extract_images=False, callback=False)
    elif entry == "extract_text_and_images":
        _child_extract_text_and_images(tag, value)
    else:
        raise SystemExit(f"unknown entry {entry}")


@pytest.mark.parametrize(
    ("tag", "value"),
    [
        (_DNG_PRIVATE_DATA, _COUNT),
        (_ACTIVE_AREA, _ACTIVE_AREA_VALUE),
    ],
)
def test_read_pdf_file_resave_does_not_crash_or_disclose(tag: int, value: int) -> None:
    completed = _run_child("read_pdf", tag, value)
    _assert_safe(completed)


@pytest.mark.parametrize(
    ("tag", "value"),
    [
        (_DNG_PRIVATE_DATA, _COUNT),
        (_ACTIVE_AREA, _ACTIVE_AREA_VALUE),
    ],
)
def test_image_callback_resave_does_not_crash_or_disclose(tag: int, value: int) -> None:
    completed = _run_child("read_pdf_callback", tag, value)
    _assert_safe(completed)


@pytest.mark.parametrize(
    ("tag", "value"),
    [
        (_DNG_PRIVATE_DATA, _COUNT),
        (_ACTIVE_AREA, _ACTIVE_AREA_VALUE),
    ],
)
def test_extract_text_and_images_resave_does_not_crash_or_disclose(
    tag: int, value: int
) -> None:
    completed = _run_child("extract_text_and_images", tag, value)
    _assert_safe(completed)


def test_benign_tiff_tag_still_round_trips() -> None:
    """A tag libtiff does not define is still extracted. This keeps the
    re-save path under test: a change that skips every image would not
    satisfy the cases above by accident only if this one still sees a tag.
    """
    completed = _run_child("read_pdf", _BENIGN_TAG, _COUNT)
    result = _assert_safe(completed)
    images = result["images"]
    assert len(images) == 1
    assert images[0]["format"] == "TIFF"
    assert images[0]["tag"] == {"type": "int", "value": _COUNT}


def test_extraction_disabled_does_not_touch_the_encoder() -> None:
    completed = _run_child("read_pdf_off", _DNG_PRIVATE_DATA, _COUNT)
    result = _assert_safe(completed)
    assert result["images"] == []


if __name__ == "__main__":
    _main()
