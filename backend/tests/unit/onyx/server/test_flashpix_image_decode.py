"""User-supplied images must not abort the process while Pillow decodes them.

Pillow 12.1.1, the version locked in ``uv.lock``, decodes a FlashPix JPEG tile
whose raw mode is ``LA`` (two bytes per pixel) by asking libjpeg for
``JCS_UNKNOWN``. A JPEG with four components is then copied, four bytes per
pixel, into that two-byte tile buffer (``null_convert`` in libjpeg-turbo). The
upload and oversized-image paths below both reach that decoder. These tests
run the product code in a child process and fail while that write aborts the
child. Pillow 13.0.0 rejects the same tile with ``OSError`` and the child
exits normally.
"""

from __future__ import annotations

import gc
import os
import struct
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile
from PIL import Image
from PIL import UnidentifiedImageError

_BACKEND_ROOT = Path(__file__).resolve().parents[4]
_SECTOR = 512
_FREESECT = 0xFFFFFFFF
_ENDOFCHAIN = 0xFFFFFFFE
_FATSECT = 0xFFFFFFFD
_NOSTREAM = 0xFFFFFFFF
_FPX_CLSID = struct.pack("<IHH", 0x56616700, 0xC154, 0x11CE) + bytes.fromhex("855300AA00A1F95B")


def _pad(data: bytes, size: int) -> bytes:
    if len(data) >= size:
        return data
    return data + b"\x00" * (size - len(data))


def _dir_entry(
    name: str,
    entry_type: int,
    *,
    left: int = _NOSTREAM,
    right: int = _NOSTREAM,
    child: int = _NOSTREAM,
    clsid: bytes = b"\x00" * 16,
    start: int = _ENDOFCHAIN,
    size: int = 0,
) -> bytes:
    raw = name.encode("utf-16le") + b"\x00\x00"
    name_buf = raw + b"\x00" * (64 - len(raw))
    return struct.pack(
        "<64sHBBIII16sIQQIII",
        name_buf,
        len(raw),
        entry_type,
        1,
        left,
        right,
        child,
        clsid,
        0,
        0,
        0,
        start & 0xFFFFFFFF,
        size & 0xFFFFFFFF,
        0,
    )


def _crafted_flashpix() -> bytes:
    """OLE2 FlashPix whose only tile is a 4-component JPEG in raw mode ``LA``.

    The subimage header is the layout ``FpxImagePlugin._open_subimage`` reads:
    a 28-byte prefix, a 36-byte descriptor, then 16-byte tile records.
    Compression 2 selects the JPEG decoder with raw mode ``LA``.
    """

    jpeg_buf = BytesIO()
    # Wide and short so the compressed file stays inside one FAT sector table,
    # while each decoded scanline is far larger than the 2-byte LA tile buffer.
    Image.new("CMYK", (8192, 2), (8, 16, 24, 32)).save(jpeg_buf, format="JPEG", quality=20)
    jpeg = jpeg_buf.getvalue()

    color_blob = struct.pack("<IIII", 0, 2, 0x00018000, 0x00017FFE)
    properties = [
        (0x1000002, struct.pack("<II", 3, 64)),
        (0x1000003, struct.pack("<II", 3, 64)),
        (0x2000002, struct.pack("<II", 65, len(color_blob)) + color_blob),
    ]
    value_base = 8 + len(properties) * 8
    entries = b""
    values = b""
    for property_id, value in properties:
        entries += struct.pack("<II", property_id, value_base + len(values))
        values += value
    section_body = struct.pack("<II", 0, len(properties)) + entries + values
    section = struct.pack("<I", len(section_body)) + section_body[4:]
    property_header = struct.pack("<HHII", 0xFFFE, 0, 0x00020001, 0) + b"\x00" * 16
    property_fmt = b"\x00" * 16 + struct.pack("<I", 48)
    property_stream = _pad(property_header + property_fmt + section, 4096)

    prefix = bytes.fromhex("feff0000030000000000010054c1ce11855300aa00a1f95b01000000")
    # width, height, tile count, tile width, tile height, channels,
    # byte offset of the tile table, tile-record length.
    info = struct.pack("<9i", 0x24, 64, 64, 1, 64, 64, 1, 36, 16)
    # Data offset 0 is read as 28 by the plugin. Compression 2, no
    # internal color conversion, no external JPEG tables.
    tile = struct.pack("<III", 0, 0x1000, 2) + b"\x00" * 4
    header_stream = _pad(prefix + info + tile, 4096)
    data_stream = _pad(b"\x00" * 28 + jpeg, 4096)

    sectors: list[bytearray] = [bytearray(_SECTOR)]

    def add(data: bytes) -> tuple[int, int]:
        remainder = len(data) % _SECTOR
        padded = data if remainder == 0 else data + b"\x00" * (_SECTOR - remainder)
        start = len(sectors)
        for offset in range(0, len(padded), _SECTOR):
            sectors.append(bytearray(padded[offset : offset + _SECTOR]))
        return start, len(data)

    property_start, property_size = add(property_stream)
    header_start, header_size = add(header_stream)
    data_start, data_size = add(data_stream)
    directory_start = len(sectors)
    sectors.append(bytearray(_SECTOR))
    sectors.append(bytearray(_SECTOR))
    if len(sectors) > 128:
        raise RuntimeError("crafted image does not fit in one FAT sector")

    directory = b"".join(
        [
            _dir_entry("Root Entry", 5, child=1, clsid=_FPX_CLSID),
            _dir_entry("Data Object Store 000001", 1, child=2),
            _dir_entry(
                "\x05Image Contents",
                2,
                right=3,
                start=property_start,
                size=property_size,
            ),
            _dir_entry("Resolution 0000", 1, child=4),
            _dir_entry(
                "Subimage 0000 Header",
                2,
                right=5,
                start=header_start,
                size=header_size,
            ),
            _dir_entry(
                "Subimage 0000 Data",
                2,
                start=data_start,
                size=data_size,
            ),
        ]
    )
    directory = directory + b"\x00" * (1024 - len(directory))
    sectors[directory_start] = bytearray(directory[:_SECTOR])
    sectors[directory_start + 1] = bytearray(directory[_SECTOR:])

    fat = [_FREESECT] * 128
    fat[0] = _FATSECT

    def chain(start: int, size: int) -> None:
        count = (size + _SECTOR - 1) // _SECTOR
        for index in range(count - 1):
            fat[start + index] = start + index + 1
        fat[start + count - 1] = _ENDOFCHAIN

    chain(property_start, property_size)
    chain(header_start, header_size)
    chain(data_start, data_size)
    chain(directory_start, 1024)
    sectors[0] = bytearray(struct.pack("<128I", *fat))

    header = struct.pack(
        "<8s16sHHHHHHLLLLLLLLLL",
        bytes.fromhex("D0CF11E0A1B11AE1"),
        b"\x00" * 16,
        0x003E,
        3,
        0xFFFE,
        9,
        6,
        0,
        0,
        0,
        1,
        directory_start,
        0,
        0x1000,
        _ENDOFCHAIN,
        0,
        _ENDOFCHAIN,
        0,
    )
    difat = struct.pack("<I", 0) + struct.pack("<I", _FREESECT) * 108
    return header + difat + b"".join(sectors)


def _require_flashpix_tile(payload: bytes) -> None:
    with Image.open(BytesIO(payload)) as image:
        if image.format != "FPX" or not image.tile:
            raise AssertionError(f"fixture was not opened as a FlashPix tile: {image.format} {image.tile}")
        codec_name = image.tile[0][0]
        if codec_name != "jpeg":
            raise AssertionError(f"fixture tile codec is {codec_name}, expected jpeg")


def _child_upload(extension: str) -> None:
    from onyx.server.features.projects import projects_file_utils as utils
    from onyx.server.settings.models import Settings

    payload = _crafted_flashpix()
    _require_flashpix_tile(payload)

    # Settings and the default model are looked up before the image is opened.
    # Substituting them keeps the test off the database; the image bytes are
    # still decoded by Pillow.
    setattr(utils, "fetch_default_llm_model", lambda _db: None)
    setattr(utils, "get_tokenizer", lambda **_kwargs: MagicMock())
    setattr(
        utils,
        "load_settings",
        lambda: Settings(
            user_file_max_upload_size_mb=100,
            file_token_count_threshold_k=100,
        ),
    )

    upload = UploadFile(
        filename=f"photo{extension}",
        file=BytesIO(payload),
        size=len(payload),
    )
    try:
        utils.estimate_image_tokens_for_upload(upload)
    except (UnidentifiedImageError, OSError, ValueError):
        upload.file.seek(0)
    upload.file.seek(0)
    utils.categorize_uploaded_files([upload], MagicMock())
    gc.collect()
    print("safe", flush=True)


def _child_resize() -> None:
    from onyx.file_processing.image_summarization import _resize_image_if_needed

    payload = _crafted_flashpix()
    # Indexing only opens an image with Pillow when it is larger than 20MB.
    padded = _pad(payload, 20 * 1024 * 1024 + 1)
    _require_flashpix_tile(padded)
    try:
        _resize_image_if_needed(padded)
    except (UnidentifiedImageError, OSError, ValueError):
        pass
    gc.collect()
    print("safe", flush=True)


def _run_child(*args: str) -> subprocess.CompletedProcess[bytes]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{_BACKEND_ROOT}{os.pathsep}{existing}" if existing else str(_BACKEND_ROOT)
    return subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), *args],
        cwd=_BACKEND_ROOT,
        env=env,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _assert_child_survives(completed: subprocess.CompletedProcess[bytes]) -> None:
    stdout = completed.stdout.decode(errors="replace")
    stderr = completed.stderr.decode(errors="replace")
    assert completed.returncode == 0, (
        f"decoding the uploaded FlashPix image aborted the process (exit {completed.returncode})\n{stdout}\n{stderr}"
    )
    assert "safe" in stdout, stdout


@pytest.mark.parametrize("extension", [".jpg", ".jpeg", ".png", ".webp"])
def test_user_image_upload_flashpix_tile_does_not_abort(extension: str) -> None:
    _assert_child_survives(_run_child("upload", extension))


def test_oversized_image_resize_flashpix_tile_does_not_abort() -> None:
    _assert_child_survives(_run_child("resize"))


if __name__ == "__main__":
    command = sys.argv[1]
    if command == "upload":
        _child_upload(sys.argv[2])
    elif command == "resize":
        _child_resize()
    else:
        raise SystemExit(f"unknown child command {command}")
