"""Attacker-controlled 8-bit lossless (SOF3) JPEGs must not be draft-decoded.

Pillow 12.1.1 sizes the JPEG scanline buffer from the geometry requested by
``Image.thumbnail`` / ``Image.draft``. libjpeg-turbo 3.x, which the Pillow
12.1.1 wheels are built against, ignores that scale for 8-bit SOF3 input and
writes a full-width line into the smaller buffer. Pillow 13.0.0 rejects the
line before the write and raises ``OSError``.

These tests fail while that Pillow is installed: the process is aborted inside
``thumbnail``. They pass once decoding is refused before the oversized write.

Call sites that request the same downscale are covered here:

* ``_resize_image_if_needed`` (image summarization, files larger than 20 MB)
* ``process_image_sections`` (indexing of an uploaded or connected image)
* ``OnyxRuntime.get_emailable_logo`` (workspace logo used in email)

``pptx`` thumbnail grids are not covered. That script thumbnails JPEGs produced
by ``pdftoppm`` after LibreOffice conversion, not the bytes from the uploaded
file. ``Image.open`` paths that load a JPEG at full size, without ``draft`` or
``thumbnail``, are not this bug.
"""

from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from onyx.connectors.models import Document
from onyx.connectors.models import DocumentSource
from onyx.connectors.models import ImageSection
from onyx.file_processing.image_summarization import _resize_image_if_needed
from onyx.indexing.indexing_pipeline import process_image_sections
from onyx.server.runtime.onyx_runtime import OnyxRuntime
from onyx.utils.file import FileWithMimeType

# cjpeg -lossless 1 -grayscale (libjpeg-turbo 3.1.3) of an 8192x8192 zero PGM.
# SOF3 precision is 8, and both sides are 0x2000. The entropy scan is the
# encoder's constant-pixel bitstream: two coded bytes, zero fill, then 0x7F
# and EOI. 8192 / 2048 == 4, and thumbnail((1024, 1024)) drafts at
# reducing_gap 2, so the requested scale is 4.
_SOF3_HEADER = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000"
    "ffc3000b082000200001011100"
    "ffc4001500010100000000000000000000000000000008"
    "ffda00080101000100009fc0"
)
_SOF3_TAIL = bytes.fromhex("7fffd9")
_SOF3_FILE_LENGTH = 8_388_678
_RESIZE_THRESHOLD_BYTES = 20 * 1024 * 1024


def _lossless_sof3_grayscale() -> bytes:
    zeros = _SOF3_FILE_LENGTH - len(_SOF3_HEADER) - len(_SOF3_TAIL)
    image = _SOF3_HEADER + (b"\x00" * zeros) + _SOF3_TAIL
    assert b"\xff\xc3" in image
    return image


def _over_resize_threshold(image: bytes) -> bytes:
    """Trailing bytes keep the JPEG intact and cross the 20 MB resize gate."""
    minimum = _RESIZE_THRESHOLD_BYTES + 1
    if len(image) >= minimum:
        return image
    return image + (b"\x00" * (minimum - len(image)))


def test_resize_image_if_needed_rejects_oversized_sof3_jpeg() -> None:
    """Summarization resize must refuse the SOF3 image instead of downscaling it."""
    payload = _over_resize_threshold(_lossless_sof3_grayscale())
    assert len(payload) > _RESIZE_THRESHOLD_BYTES

    with pytest.raises(OSError):
        _resize_image_if_needed(payload)


def test_image_section_indexing_rejects_oversized_sof3_jpeg() -> None:
    """Indexing an uploaded image must not summarize a SOF3 file via thumbnail."""
    payload = _over_resize_threshold(_lossless_sof3_grayscale())
    record = MagicMock()
    record.display_name = "upload.jpg"
    file_store = MagicMock()
    file_store.read_file_record.return_value = record
    file_store.read_file.return_value = BytesIO(payload)

    llm = MagicMock()
    response = MagicMock()
    response.choice.message.content = "summarized-the-image"
    llm.invoke.return_value = response

    document = Document(
        id="user-upload",
        sections=[ImageSection(image_file_id="stored-image", link=None)],
        source=DocumentSource.FILE,
        semantic_identifier="upload.jpg",
        metadata={},
    )

    with (
        patch(
            "onyx.indexing.indexing_pipeline.get_image_extraction_and_analysis_enabled",
            return_value=True,
        ),
        patch(
            "onyx.indexing.indexing_pipeline.get_default_llm_with_vision",
            return_value=llm,
        ),
        patch(
            "onyx.indexing.indexing_pipeline.get_default_file_store",
            return_value=file_store,
        ),
    ):
        indexed = process_image_sections([document])

    assert llm.invoke.call_count == 0
    assert indexed[0].processed_sections[0].text == "[Error processing image]"
    assert indexed[0].processed_sections[0].image_file_id == "stored-image"


def test_emailable_logo_rejects_sof3_jpeg() -> None:
    """A JPEG workspace logo is always passed to thumbnail before emailing."""
    logo = FileWithMimeType(
        data=_lossless_sof3_grayscale(),
        mime_type="image/jpeg",
    )

    with (
        patch.object(OnyxRuntime, "get_logo", return_value=logo),
        pytest.raises(OSError),
    ):
        OnyxRuntime.get_emailable_logo()
