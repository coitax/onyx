"""Upload-time archive extraction must refuse decompression bombs.

The project upload path (``POST /api/user/projects/file/upload`` →
``categorize_uploaded_files``) only compares the compressed upload against
``user_file_max_upload_size_mb``. EPUB, DOCX, PPTX, and XLSX are ZIP archives.
Their extractors then decompress every entry they parse, with no cap on the
declared uncompressed size, the compression ratio, or the entry count.

These tests are the contract for a fix. They fail while that cap is missing
and pass once extraction refuses the archive before the oversized entry is
materialized. Reading the central directory and then still decompressing the
entry does not pass: the byte counter wraps ``zipfile.ZipExtFile.read``, which
is what delivers decompressed entry bytes to the extractors.

Required limits, measured from the ZIP central directory before entry bodies
are read:

- total declared uncompressed size at most ``MAX_ARCHIVE_UNCOMPRESSED_BYTES``
- a single entry's declared size at most that same cap
- every entry's compression ratio (its declared uncompressed size divided
  by its compressed size) at most ``MAX_ARCHIVE_COMPRESSION_RATIO``.
  The archive-wide ratio is not enough: a PPTX or XLSX package carries
  other parts, so one huge text part can sit under a modest total ratio.
- entry count at most ``MAX_ARCHIVE_ENTRY_COUNT``

An archive that exceeds any of those must be refused. Refusing means the
extractor raises, or returns an empty string, and ``ZipExtFile.read`` has
delivered at most ``MAX_REJECTED_BODY_BYTES`` of decompressed body data.
That allowance covers small package metadata such as ``[Content_Types].xml``.
It is smaller than every bomb payload below, so a full read of the payload
fails the test.

Operations covered:

- ``categorize_uploaded_files`` (the upload request)
- ``extract_file_text`` (upload, chat file load, file-reader tool)
- ``extract_text_and_images`` (indexing). This function swallows extractor
  errors, so a refusal that happens only after the entry was inflated still
  fails the byte counter.
- ``epub_to_text``, ``read_docx_file``, ``pptx_to_text``, ``xlsx_to_text``
  (the functions the paths above call). ``upload_files`` also calls
  ``read_docx_file`` directly when the part's content type is a DOCX.

Left untested, on purpose:

- ``upload_files`` with ``unzip=True`` fully reads generic ``.zip`` members.
  That endpoint is curator/admin, and a ``.zip`` submitted to the project
  upload is rejected as an unsupported type before ``upload_files`` runs.
- User-library ``upload_zip`` checks the declared size against a multi-gigabyte
  storage quota and then reads each member. It is a different limit, not this
  text-extraction path.
- Connector sync calls the same four extractors. Those calls are covered by
  the extractor tests, not by a second upload request.
"""

from __future__ import annotations

import zipfile
from collections.abc import Callable
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from fastapi import UploadFile

from onyx.file_processing.extract_file_text import epub_to_text
from onyx.file_processing.extract_file_text import extract_file_text
from onyx.file_processing.extract_file_text import extract_text_and_images
from onyx.file_processing.extract_file_text import pptx_to_text
from onyx.file_processing.extract_file_text import read_docx_file
from onyx.file_processing.extract_file_text import xlsx_to_text
from onyx.natural_language_processing.utils import BaseTokenizer
from onyx.server.features.projects import projects_file_utils as upload_utils
from onyx.server.settings.models import Settings

# Contract. See the module docstring.
MAX_ARCHIVE_UNCOMPRESSED_BYTES = 2 * 1024 * 1024
MAX_ARCHIVE_COMPRESSION_RATIO = 100
MAX_ARCHIVE_ENTRY_COUNT = 1000
MAX_REJECTED_BODY_BYTES = 64 * 1024

_KINDS = ("epub", "docx", "pptx", "xlsx")
_MARKER = b"hello-onyx"
_BOMB_BYTE = b"A"

_XLSX_TEMPLATE: bytes | None = None
_PPTX_TEMPLATE: bytes | None = None


class _Tokenizer(BaseTokenizer):
    """Counts one token per call. Avoids materializing a token per character."""

    def encode(self, text: str) -> list[int]:
        return [1] if text else []

    def tokenize(self, text: str) -> list[str]:
        return [text] if text else []

    def decode(self, _tokens: list[int]) -> str:
        return ""


def _settings() -> Settings:
    # 0 disables the token cap, so a bomb is not rejected merely for being long
    # after it has already been decompressed. 100 MB is the product default
    # and is far above every compressed fixture in this module.
    return Settings(
        user_file_max_upload_size_mb=100,
        file_token_count_threshold_k=0,
    )


@pytest.fixture
def decompressed_bytes(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Count bytes returned by ``ZipExtFile.read`` (decompressed entry bodies)."""
    counter = {"n": 0}
    original = zipfile.ZipExtFile.read

    def counting_read(self: zipfile.ZipExtFile, n: int = -1) -> bytes:
        data = original(self, n)
        counter["n"] += len(data)
        return data

    monkeypatch.setattr(zipfile.ZipExtFile, "read", counting_read)
    return counter


@pytest.fixture(autouse=True)
def _local_extraction(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the local extractor branch.

    ``get_unstructured_api_key`` returns None when the key has never been
    stored (``KvKeyNotFoundError``). That is the branch that decompresses
    archives inside the API process. The key-value store is not running here,
    so the lookup is stubbed to that same result.
    """
    monkeypatch.setattr(
        "onyx.file_processing.extract_file_text.get_unstructured_api_key",
        lambda: None,
    )


@pytest.fixture
def patched_upload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(upload_utils, "fetch_default_llm_model", lambda _db: None)
    monkeypatch.setattr(upload_utils, "get_tokenizer", lambda **_kwargs: _Tokenizer())
    monkeypatch.setattr(upload_utils, "load_settings", _settings)


def _zip(entries: dict[str, bytes], compression: int) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data, compress_type=compression)
    return buffer.getvalue()


def _rewrite(
    template: bytes,
    replacements: dict[str, bytes],
    compression: int,
    extra: dict[str, bytes] | None = None,
) -> bytes:
    buffer = BytesIO()
    with (
        zipfile.ZipFile(BytesIO(template)) as source,
        zipfile.ZipFile(buffer, "w") as output,
    ):
        for info in source.infolist():
            payload = replacements.get(info.filename)
            if payload is None:
                payload = source.read(info.filename)
            output.writestr(info.filename, payload, compress_type=compression)
        for name, payload in (extra or {}).items():
            output.writestr(name, payload, compress_type=compression)
    return buffer.getvalue()


def _xlsx_template() -> bytes:
    global _XLSX_TEMPLATE
    if _XLSX_TEMPLATE is None:
        from openpyxl import Workbook

        workbook = Workbook()
        workbook.active["A1"] = _MARKER.decode()
        buffer = BytesIO()
        workbook.save(buffer)
        _XLSX_TEMPLATE = buffer.getvalue()
    return _XLSX_TEMPLATE


def _pptx_template() -> bytes:
    global _PPTX_TEMPLATE
    if _PPTX_TEMPLATE is None:
        from pptx import Presentation

        presentation = Presentation()
        slide = presentation.slides.add_slide(presentation.slide_layouts[1])
        slide.shapes.title.text = _MARKER.decode()
        buffer = BytesIO()
        presentation.save(buffer)
        _PPTX_TEMPLATE = buffer.getvalue()
    return _PPTX_TEMPLATE


def _docx_entries(text: bytes) -> dict[str, bytes]:
    content_types = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>"""
    relationships = b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document = (
        b"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body><w:p><w:r><w:t>"""
        + text
        + b"""</w:t></w:r></w:p></w:body>
</w:document>"""
    )
    return {
        "[Content_Types].xml": content_types,
        "_rels/.rels": relationships,
        "word/document.xml": document,
    }


def _sheet_with_text(text: bytes) -> bytes:
    with zipfile.ZipFile(BytesIO(_xlsx_template())) as archive:
        sheet = archive.read("xl/worksheets/sheet1.xml")
    if sheet.count(_MARKER) != 1:
        raise AssertionError("xlsx template no longer contains exactly one marker")
    return sheet.replace(_MARKER, text)


def _slide_with_text(text: bytes) -> bytes:
    with zipfile.ZipFile(BytesIO(_pptx_template())) as archive:
        slide = archive.read("ppt/slides/slide1.xml")
    if slide.count(_MARKER) != 1:
        raise AssertionError("pptx template no longer contains exactly one marker")
    return slide.replace(_MARKER, text)


def _build(kind: str, text: bytes, compression: int) -> bytes:
    if kind == "epub":
        return _zip(
            {"chapter.xhtml": b"<html><body>" + text + b"</body></html>"},
            compression,
        )
    if kind == "docx":
        return _zip(_docx_entries(text), compression)
    if kind == "xlsx":
        return _rewrite(
            _xlsx_template(),
            {"xl/worksheets/sheet1.xml": _sheet_with_text(text)},
            compression,
        )
    if kind == "pptx":
        return _rewrite(
            _pptx_template(),
            {"ppt/slides/slide1.xml": _slide_with_text(text)},
            compression,
        )
    raise AssertionError(kind)


def _build_many_entries(kind: str) -> bytes:
    """Valid document plus enough extra members to exceed the entry cap.

    Each extra member is a distinct incompressible blob so the archive stays
    under the uncompressed-size cap and under the ratio cap. Only an entry
    count check refuses it.
    """
    extra_count = MAX_ARCHIVE_ENTRY_COUNT + 1
    extra: dict[str, bytes] = {}
    for index in range(extra_count):
        blob = bytes((index + offset) % 251 for offset in range(180))
        extra[f"extra/item-{index}.bin"] = blob
    benign = _build(kind, _MARKER, zipfile.ZIP_STORED)
    return _rewrite(benign, {}, zipfile.ZIP_STORED, extra=extra)


def _stats(archive: bytes) -> tuple[int, int, int, float]:
    """Return entry count, total sizes, and the largest per-entry ratio.

    The ratio is per entry. Package parts that do not compress well (themes,
    thumbnails) would hide a bomb if only the archive-wide ratio were used.
    """
    with zipfile.ZipFile(BytesIO(archive)) as zip_file:
        members = [info for info in zip_file.infolist() if not info.is_dir()]
    uncompressed = sum(info.file_size for info in members)
    compressed = sum(info.compress_size for info in members)
    max_ratio = max(
        (info.file_size / max(info.compress_size, 1) for info in members),
        default=0.0,
    )
    return len(members), uncompressed, compressed, max_ratio


def _call(func: Callable[[], str]) -> str | Exception:
    try:
        return func()
    except Exception as exc:
        return exc


def _extract_direct(kind: str, archive: bytes) -> str:
    stream = BytesIO(archive)
    name = f"sample.{kind}"
    if kind == "epub":
        return epub_to_text(stream)
    if kind == "docx":
        text, _images = read_docx_file(stream, name)
        return text
    if kind == "pptx":
        return pptx_to_text(stream, name)
    if kind == "xlsx":
        return xlsx_to_text(stream, name)
    raise AssertionError(kind)


def _assert_refused_without_inflating(
    counter: dict[str, int],
    outcome: str | Exception,
) -> None:
    assert counter["n"] <= MAX_REJECTED_BODY_BYTES, (
        f"decompressed {counter['n']} entry bytes before refusing; "
        f"the limit for an over-cap archive is {MAX_REJECTED_BODY_BYTES}. "
        "Checking the central directory and then still reading the entry "
        "does not satisfy this."
    )
    if isinstance(outcome, Exception):
        return
    assert (
        len(outcome) <= MAX_REJECTED_BODY_BYTES
    ), f"extractor returned {len(outcome)} characters from an over-cap archive"
    assert _BOMB_BYTE * 1024 not in outcome.encode(
        "utf-8", errors="ignore"
    ), "extractor returned the decompression-bomb payload"
    # A successful extraction of a benign marker means the archive was accepted.
    # Over-cap archives must not produce document text.
    assert (
        _MARKER.decode() not in outcome
    ), "extractor returned document text for an archive past the decompression limits"


def _upload(name: str, archive: bytes) -> UploadFile:
    return UploadFile(filename=name, file=BytesIO(archive), size=len(archive))


@pytest.mark.parametrize("kind", _KINDS)
def test_benign_archive_extracts_marker(kind: str) -> None:
    """Small archives still have to extract. A fix must not reject every ZIP."""
    archive = _build(kind, _MARKER, zipfile.ZIP_DEFLATED)
    text = _extract_direct(kind, archive)
    assert _MARKER.decode() in text


@pytest.mark.usefixtures("patched_upload")
@pytest.mark.parametrize("kind", _KINDS)
def test_benign_archive_is_accepted_on_upload(kind: str) -> None:
    archive = _build(kind, _MARKER, zipfile.ZIP_DEFLATED)
    upload = _upload(f"notes.{kind}", archive)
    result = upload_utils.categorize_uploaded_files([upload], MagicMock())
    assert result.rejected == []
    assert result.acceptable == [upload]
    assert result.acceptable_file_to_token_count[f"notes.{kind}"] >= 0


@pytest.mark.parametrize("kind", _KINDS)
def test_high_ratio_archive_is_not_inflated(
    kind: str,
    decompressed_bytes: dict[str, int],
) -> None:
    """Repeated bytes compress far below the upload cap and expand past it.

    The uncompressed size here is under ``MAX_ARCHIVE_UNCOMPRESSED_BYTES``,
    so a check of absolute size alone still inflates the entry. The ratio
    check has to refuse it first.
    """
    archive = _build(kind, _BOMB_BYTE * (256 * 1024), zipfile.ZIP_DEFLATED)
    count, uncompressed, compressed, ratio = _stats(archive)
    assert count <= MAX_ARCHIVE_ENTRY_COUNT
    assert uncompressed <= MAX_ARCHIVE_UNCOMPRESSED_BYTES
    assert compressed < 1024 * 1024
    assert ratio > MAX_ARCHIVE_COMPRESSION_RATIO
    assert len(archive) < 1024 * 1024

    decompressed_bytes["n"] = 0
    outcome = _call(lambda: _extract_direct(kind, archive))
    _assert_refused_without_inflating(decompressed_bytes, outcome)


@pytest.mark.parametrize("kind", _KINDS)
def test_high_ratio_archive_extract_file_text_is_not_inflated(
    kind: str,
    decompressed_bytes: dict[str, int],
) -> None:
    """``extract_file_text(..., break_on_unprocessable=False)`` is the upload call.

    Exceptions raised by an extractor are swallowed there and become ``""``.
    Swallowing after a full inflate still fails.
    """
    archive = _build(kind, _BOMB_BYTE * (256 * 1024), zipfile.ZIP_DEFLATED)
    decompressed_bytes["n"] = 0
    outcome = _call(
        lambda: extract_file_text(
            BytesIO(archive),
            f"sample.{kind}",
            break_on_unprocessable=False,
        )
    )
    _assert_refused_without_inflating(decompressed_bytes, outcome)


@pytest.mark.parametrize("kind", _KINDS)
def test_high_ratio_archive_indexing_extract_is_not_inflated(
    kind: str,
    decompressed_bytes: dict[str, int],
) -> None:
    """Indexing calls ``extract_text_and_images``, which swallows errors."""
    archive = _build(kind, _BOMB_BYTE * (256 * 1024), zipfile.ZIP_DEFLATED)
    decompressed_bytes["n"] = 0

    def _run() -> str:
        result = extract_text_and_images(BytesIO(archive), f"sample.{kind}")
        return result.text_content

    outcome = _call(_run)
    _assert_refused_without_inflating(decompressed_bytes, outcome)


@pytest.mark.usefixtures("patched_upload")
@pytest.mark.parametrize("kind", _KINDS)
def test_high_ratio_upload_is_rejected_without_inflating(
    kind: str,
    decompressed_bytes: dict[str, int],
) -> None:
    """The project-upload request: compressed size passes, extraction must not."""
    archive = _build(kind, _BOMB_BYTE * (256 * 1024), zipfile.ZIP_DEFLATED)
    _count, uncompressed, _compressed, max_entry_ratio = _stats(archive)
    assert len(archive) < 100 * 1024 * 1024
    assert max_entry_ratio > MAX_ARCHIVE_COMPRESSION_RATIO
    assert uncompressed > MAX_REJECTED_BODY_BYTES

    upload = _upload(f"bomb.{kind}", archive)
    decompressed_bytes["n"] = 0
    categorized = upload_utils.categorize_uploaded_files([upload], MagicMock())

    assert decompressed_bytes["n"] <= MAX_REJECTED_BODY_BYTES, (
        f"upload categorization decompressed {decompressed_bytes['n']} bytes "
        f"of a {len(archive)}-byte file (declared uncompressed {uncompressed}, "
        f"max entry ratio {max_entry_ratio:.0f}). "
        "The compressed-size check did not stop it."
    )
    assert categorized.acceptable == []
    assert categorized.rejected
    assert categorized.rejected[0].filename == f"bomb.{kind}"


@pytest.mark.parametrize("kind", _KINDS)
def test_stored_archive_over_uncompressed_cap_is_not_inflated(
    kind: str,
    decompressed_bytes: dict[str, int],
) -> None:
    """Uncompressed (stored) entries have a ratio near 1.

    A ratio check alone does not refuse them. The declared-size check has to,
    and it has to happen before the entry body is read. The stored file is
    still under the 100 MB upload limit.
    """
    archive = _build(
        kind,
        _BOMB_BYTE * (MAX_ARCHIVE_UNCOMPRESSED_BYTES + 4096),
        zipfile.ZIP_STORED,
    )
    _count, uncompressed, _compressed, ratio = _stats(archive)
    assert uncompressed > MAX_ARCHIVE_UNCOMPRESSED_BYTES
    assert ratio < MAX_ARCHIVE_COMPRESSION_RATIO
    assert len(archive) < 100 * 1024 * 1024

    decompressed_bytes["n"] = 0
    outcome = _call(lambda: _extract_direct(kind, archive))
    _assert_refused_without_inflating(decompressed_bytes, outcome)


@pytest.mark.parametrize("kind", _KINDS)
def test_too_many_zip_entries_are_refused_without_reading_them_all(
    kind: str,
    decompressed_bytes: dict[str, int],
) -> None:
    archive = _build_many_entries(kind)
    count, uncompressed, _compressed, ratio = _stats(archive)
    assert count > MAX_ARCHIVE_ENTRY_COUNT
    assert uncompressed <= MAX_ARCHIVE_UNCOMPRESSED_BYTES
    assert ratio < MAX_ARCHIVE_COMPRESSION_RATIO

    decompressed_bytes["n"] = 0
    outcome = _call(lambda: _extract_direct(kind, archive))
    _assert_refused_without_inflating(decompressed_bytes, outcome)
