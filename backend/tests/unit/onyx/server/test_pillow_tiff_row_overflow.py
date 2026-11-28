"""Attacker-controlled TIFFs must not be decoded by the project upload or PDF paths.

Pillow 12.1.1 reads past the row libtiff produced when a tiled TIFF repeats
BitsPerSample, or when PhotometricInterpretation is repeated so Pillow's
unpacker is wider than the RGBA buffer. These tests fail while that Pillow is
installed. They pass once those decodes are refused before the copy.

Run:
    PYTHONPATH=backend pytest -xv backend/tests/unit/onyx/server/test_pillow_tiff_row_overflow.py
"""

from __future__ import annotations

import io
import os
import struct
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile
from PIL import Image

from onyx.file_processing.extract_file_text import read_pdf_file
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.server.features.projects import projects_file_utils as utils
from onyx.server.settings.models import Settings

_REPO_ROOT = Path(__file__).resolve().parents[5]
_BACKEND_ROOT = _REPO_ROOT / "backend"

# Wider than the mapped heap after the app imports, still under the 12000*12000
# pixel cap projects_file_utils installs before it decodes an upload.
_WIDE_TILE_WIDTH = 40_000_000

_SHORT = 3
_LONG = 4


class _Tokenizer(BaseTokenizer):
    def encode(self, text: str) -> list[int]:
        return [1] * len(text)

    def tokenize(self, text: str) -> list[str]:
        return list(text)

    def decode(self, _tokens: list[int]) -> str:
        return ""


def _u16(value: int) -> bytes:
    return struct.pack("<H", value)


def _u16s(*values: int) -> bytes:
    return b"".join(struct.pack("<H", value) for value in values)


def _u32(value: int) -> bytes:
    return struct.pack("<I", value)


def _packbits_literals(data: bytes) -> bytes:
    encoded = bytearray()
    offset = 0
    while offset < len(data):
        count = min(128, len(data) - offset)
        encoded.append(count - 1)
        encoded += data[offset : offset + count]
        offset += count
    return bytes(encoded)


def _packbits_repeat(byte: int, count: int) -> bytes:
    encoded = bytearray()
    remaining = count
    while remaining:
        run = min(128, remaining)
        encoded.append((1 - run) & 0xFF)
        encoded.append(byte)
        remaining -= run
    return bytes(encoded)


def _tiff(entries: list[tuple[int, int, int, bytes]], tile: bytes) -> bytes:
    """Little-endian TIFF. TileOffsets and TileByteCounts values of 0 are filled in."""
    count = len(entries)
    data_offset = 8 + 2 + count * 12 + 4
    if data_offset % 2:
        data_offset += 1
    cursor = data_offset
    encoded: list[tuple[int, int, int, bytes, bytes]] = []
    blobs: list[bytes] = []
    for tag, typ, n_values, raw in entries:
        if len(raw) <= 4:
            encoded.append((tag, typ, n_values, raw.ljust(4, b"\x00"), b""))
        else:
            encoded.append((tag, typ, n_values, struct.pack("<I", cursor), raw))
            blobs.append(raw)
            cursor += len(raw) + (len(raw) % 2)
            if len(raw) % 2:
                blobs.append(b"\x00")
    tile_offset = cursor
    buffer = bytearray(b"II" + struct.pack("<HI", 42, 8) + struct.pack("<H", count))
    for tag, typ, n_values, value, _raw in encoded:
        if tag == 324 and value == b"\x00\x00\x00\x00":
            value = struct.pack("<I", tile_offset)
        if tag == 325 and value == b"\x00\x00\x00\x00":
            value = struct.pack("<I", len(tile))
        buffer += struct.pack("<HHI", tag, typ, n_values) + value
    buffer += struct.pack("<I", 0)
    if len(buffer) < data_offset:
        buffer += b"\x00" * (data_offset - len(buffer))
    for blob in blobs:
        buffer += blob
    if len(buffer) < tile_offset:
        buffer += b"\x00" * (tile_offset - len(buffer))
    buffer += tile
    return bytes(buffer)


def duplicate_bits_per_sample_tiff(width: int, height: int) -> bytes:
    """Tiled PackBits RGB whose first BitsPerSample is 1 and whose last is 8,8,8.

    Pillow keeps the last tag and unpacks 3 bytes per pixel. libtiff 4.7.1 keeps
    the first tag, so the tile row is about one eighth of that.
    """
    libtiff_row = (width * 3 + 7) // 8
    tile = _packbits_repeat(0xA5, libtiff_row * height)
    entries = [
        (256, _LONG, 1, _u32(width)),
        (257, _LONG, 1, _u32(height)),
        (258, _SHORT, 1, _u16(1)),
        (258, _SHORT, 3, _u16s(8, 8, 8)),
        (259, _SHORT, 1, _u16(32773)),
        (262, _SHORT, 1, _u16(2)),
        (274, _SHORT, 1, _u16(2)),
        (277, _SHORT, 1, _u16(3)),
        (284, _SHORT, 1, _u16(1)),
        (322, _LONG, 1, _u32(width)),
        (323, _LONG, 1, _u32(height)),
        (324, _LONG, 1, b"\x00\x00\x00\x00"),
        (325, _LONG, 1, b"\x00\x00\x00\x00"),
    ]
    return _tiff(entries, tile)


def duplicate_photometric_tiff(width: int, height: int) -> bytes:
    """Tiled PackBits whose first photometric tag is YCbCr and whose last is RGB.

    Pillow selects 16-bit RGB (48 bits per pixel). libtiff follows YCbCr and the
    RGBA decoder fills 32 bits per pixel.
    """
    tile = _packbits_literals(bytes([0x11, 0x80, 0x80]) * (width * height))
    entries = [
        (256, _LONG, 1, _u32(width)),
        (257, _LONG, 1, _u32(height)),
        (258, _SHORT, 1, _u16(8)),
        (258, _SHORT, 3, _u16s(16, 16, 16)),
        (259, _SHORT, 1, _u16(32773)),
        (262, _SHORT, 1, _u16(6)),
        (262, _SHORT, 1, _u16(2)),
        (274, _SHORT, 1, _u16(2)),
        (277, _SHORT, 1, _u16(3)),
        (284, _SHORT, 1, _u16(1)),
        (322, _LONG, 1, _u32(width)),
        (323, _LONG, 1, _u32(height)),
        (324, _LONG, 1, b"\x00\x00\x00\x00"),
        (325, _LONG, 1, b"\x00\x00\x00\x00"),
        (530, _SHORT, 2, _u16s(1, 1)),
    ]
    return _tiff(entries, tile)


def _pdf_with_dct_stream(stream: bytes, width: int, height: int) -> bytes:
    """One-page PDF whose image stream is ``stream`` and whose filter is DCTDecode.

    DCTDecode returns the stream unchanged, and Image.open then sniffs the bytes.
    """
    content = f"q\n{width} 0 0 {height} 0 0 cm\n/Im1 Do\nQ\n".encode()
    image = (
        (
            f"<< /Type /XObject /Subtype /Image /Width {width} /Height {height} "
            f"/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode "
            f"/Length {len(stream)} >>\nstream\n"
        ).encode()
        + stream
        + b"\nendstream"
    )
    contents = f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream"
    parts = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 "
            + f"{width} {height}".encode()
            + b"] /Resources << /XObject << /Im1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        image,
        contents,
    ]
    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, part in enumerate(parts, start=1):
        offsets.append(len(output))
        output += f"{index} 0 obj\n".encode() + part + b"\nendobj\n"
    xref = len(output)
    output += f"xref\n0 {len(parts) + 1}\n".encode()
    output += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        output += f"{offset:010d} 00000 n \n".encode()
    output += (
        f"trailer << /Size {len(parts) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF"
    ).encode()
    return bytes(output)


def _patch_upload_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "fetch_default_llm_model", lambda _db: None)
    monkeypatch.setattr(utils, "get_tokenizer", lambda **_kwargs: _Tokenizer())
    monkeypatch.setattr(
        utils,
        "load_settings",
        lambda: Settings(
            user_file_max_upload_size_mb=50,
            file_token_count_threshold_k=100,
        ),
    )


def _upload(filename: str, payload: bytes) -> UploadFile:
    return UploadFile(filename=filename, file=io.BytesIO(payload), size=len(payload))


@pytest.mark.parametrize(
    ("builder", "width", "height"),
    [
        (duplicate_bits_per_sample_tiff, 64, 16),
        (duplicate_photometric_tiff, 32, 8),
    ],
)
def test_estimate_image_tokens_refuses_crafted_tiff(
    builder: Callable[[int, int], bytes],
    width: int,
    height: int,
) -> None:
    payload = builder(width, height)
    with pytest.raises(OSError):
        utils.estimate_image_tokens_for_upload(_upload("scan.png", payload))


@pytest.mark.parametrize("extension", [".png", ".jpg", ".jpeg", ".webp"])
@pytest.mark.parametrize(
    ("builder", "width", "height"),
    [
        (duplicate_bits_per_sample_tiff, 64, 16),
        (duplicate_photometric_tiff, 32, 8),
    ],
)
def test_project_upload_rejects_crafted_tiff_with_image_extension(
    monkeypatch: pytest.MonkeyPatch,
    extension: str,
    builder: Callable[[int, int], bytes],
    width: int,
    height: int,
) -> None:
    _patch_upload_dependencies(monkeypatch)
    payload = builder(width, height)
    result = utils.categorize_uploaded_files(
        [_upload(f"scan{extension}", payload)],
        MagicMock(),
    )

    assert result.acceptable == []
    assert [rejected.filename for rejected in result.rejected] == [f"scan{extension}"]


def test_project_upload_still_accepts_a_real_png(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_upload_dependencies(monkeypatch)
    buffer = io.BytesIO()
    Image.new("RGB", (2, 2), (10, 20, 30)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    result = utils.categorize_uploaded_files(
        [_upload("photo.png", payload)],
        MagicMock(),
    )

    assert [upload.filename for upload in result.acceptable] == ["photo.png"]
    assert result.rejected == []


def test_wide_crafted_tiff_does_not_kill_the_upload_worker() -> None:
    """A basic user's upload is decoded in-process. The worker has to stay up."""
    payload = duplicate_bits_per_sample_tiff(_WIDE_TILE_WIDTH, 1)
    child = """
import io
import sys
from unittest.mock import MagicMock
from fastapi import UploadFile
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.server.features.projects import projects_file_utils as utils
from onyx.server.settings.models import Settings

class _Tokenizer(BaseTokenizer):
    def encode(self, text):
        return [1] * len(text)
    def tokenize(self, text):
        return list(text)
    def decode(self, _tokens):
        return ""

utils.fetch_default_llm_model = lambda _db: None
utils.get_tokenizer = lambda **_kwargs: _Tokenizer()
utils.load_settings = lambda: Settings(
    user_file_max_upload_size_mb=50,
    file_token_count_threshold_k=100,
)
data = sys.stdin.buffer.read()
upload = UploadFile(filename="scan.png", file=io.BytesIO(data), size=len(data))
result = utils.categorize_uploaded_files([upload], MagicMock())
if result.acceptable:
    print("accepted")
    sys.exit(1)
print("rejected")
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_BACKEND_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", child],
        input=payload,
        capture_output=True,
        timeout=90,
        cwd=str(_REPO_ROOT),
        env=env,
        check=False,
    )

    assert completed.returncode == 0, (
        f"upload worker exited {completed.returncode}\n"
        f"stdout={completed.stdout!r}\nstderr={completed.stderr[-2000:]!r}"
    )
    assert b"rejected" in completed.stdout
    assert b"accepted" not in completed.stdout


def test_pdf_image_extraction_does_not_return_crafted_tiff_pixels() -> None:
    tiff = duplicate_bits_per_sample_tiff(64, 16)
    pdf = _pdf_with_dct_stream(tiff, 64, 16)

    _text, _metadata, images = read_pdf_file(io.BytesIO(pdf), extract_images=True)

    assert images == []
