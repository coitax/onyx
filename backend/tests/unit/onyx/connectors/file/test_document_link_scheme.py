"""Document links taken from uploaded file metadata must be http or https.

A signed-in user can upload a text file whose first line is an ONYX_METADATA
header, or a PDF whose info dictionary contains a link entry. Local file
processing copies that value onto the document section, the indexing pipeline
stores it as the chunk's source link and the document's first link, and the
chat UI later passes it to window.open. These tests follow that path and
require every stored link to use the http or https scheme.
"""

import json
import re
from io import BytesIO
from unittest.mock import MagicMock
from unittest.mock import patch

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject
from pypdf.generic import DictionaryObject
from pypdf.generic import NameObject

from onyx.connectors.file.connector import _process_file
from onyx.connectors.models import Document
from onyx.connectors.models import IndexAttemptMetadata
from onyx.indexing.chunker import Chunker
from onyx.indexing.indexing_pipeline import _upsert_documents_in_db
from onyx.indexing.indexing_pipeline import process_image_sections


_CONTROL_CHARS = re.compile(r"[\u0000-\u001F\u007F]")

# Schemes the browser treats as script or local-file navigations, including
# the characters the WHATWG URL parser ignores inside a scheme.
_UNSAFE_LINKS = (
    "javascript:void(0)",
    "JavaScript:void(0)",
    " javascript:void(0)",
    "\tjavascript:void(0)",
    "\njavascript:void(0)",
    "java\tscript:void(0)",
    "java\nscript:void(0)",
    "\x00javascript:void(0)",
    "data:text/html,hi",
    "vbscript:msgbox(1)",
    "file:///tmp/x",
)

_BODY = "Company policy text the citation quotes."


class _WordTokenizer:
    """Counts whitespace-separated words so chunking does not download a model."""

    def encode(self, string: str) -> list[int]:
        return [1] * max(1, len(string.split()))

    def tokenize(self, string: str) -> list[str]:
        parts = string.split()
        return parts or [""]


def _scheme(link: str) -> str:
    normalized = _CONTROL_CHARS.sub("", link).strip()
    if ":" not in normalized:
        return ""
    return normalized.split(":", 1)[0].lower()


def _assert_http_link(link: str | None) -> None:
    if link is None or link == "":
        return
    assert _scheme(link) in {"http", "https"}, link


def _text_upload(link: str, header: str) -> BytesIO:
    payload = json.dumps({"link": link})
    if header == "hash":
        first = f"#ONYX_METADATA={payload}\n"
    else:
        first = f"<!-- ONYX_METADATA={payload} -->\n"
    return BytesIO(f"{first}{_BODY}\n".encode())


def _pdf_upload(link: str) -> BytesIO:
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject(
        {
            NameObject("/Font"): DictionaryObject(
                {
                    NameObject("/F1"): DictionaryObject(
                        {
                            NameObject("/Type"): NameObject("/Font"),
                            NameObject("/Subtype"): NameObject("/Type1"),
                            NameObject("/BaseFont"): NameObject("/Helvetica"),
                        }
                    )
                }
            )
        }
    )
    content = DecodedStreamObject()
    content.set_data(b"BT /F1 12 Tf 72 720 Td (Company policy text) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(content)
    writer.add_metadata({"/link": link, "/Title": "notes"})
    buffer = BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    return buffer


def _index_links(documents: list[Document]) -> tuple[str, list[str]]:
    """Run the indexing steps that copy a section link into stored fields."""
    indexed = process_image_sections(documents)
    chunker = Chunker(
        tokenizer=_WordTokenizer(),
        enable_multipass=False,
        enable_large_chunks=False,
        enable_contextual_rag=False,
    )
    source_links: list[str] = []
    for chunk in chunker.chunk(indexed):
        source_links.extend(chunk.source_links.values() if chunk.source_links else [])

    captured: list = []

    def _capture_upsert(
        _db_session: object, document_metadata_batch: list, initial_boost: int = 0
    ) -> None:
        del initial_boost
        captured.extend(document_metadata_batch)

    with (
        patch(
            "onyx.indexing.indexing_pipeline.upsert_documents",
            _capture_upsert,
        ),
        patch("onyx.indexing.indexing_pipeline.upsert_document_tags"),
    ):
        _upsert_documents_in_db(
            documents,
            IndexAttemptMetadata(connector_id=1, credential_id=1),
            MagicMock(),
        )

    first_link = captured[0].first_link if captured else ""
    return first_link, source_links


def _assert_documents_are_safe(documents: list[Document]) -> None:
    assert documents, "file processing produced no document"
    saw_body = False
    for document in documents:
        for section in document.sections:
            _assert_http_link(section.link)
            if section.text and _BODY.split()[0] in section.text:
                saw_body = True
    assert saw_body, "file text was not extracted"
    first_link, source_links = _index_links(documents)
    _assert_http_link(first_link)
    for link in source_links:
        _assert_http_link(link)


def _process(
    file_name: str,
    file: BytesIO,
    file_type: str,
    metadata: dict[str, str] | None = None,
) -> list[Document]:
    # No unstructured API key is the default. The lookup raises without Postgres,
    # which aborts extraction, so substitute the missing-key result.
    with patch(
        "onyx.file_processing.extract_file_text.get_unstructured_api_key",
        return_value=None,
    ):
        return _process_file(
            file_id="user-file-1",
            file_name=file_name,
            file=file,
            metadata=metadata or {},
            pdf_pass=None,
            file_type=file_type,
        )


def test_https_file_metadata_link_is_preserved() -> None:
    link = "https://example.com/policy"
    documents = _process("notes.txt", _text_upload(link, "hash"), "text/plain")
    assert documents[0].sections[0].link == link
    first_link, source_links = _index_links(documents)
    assert first_link == link
    assert link in source_links


def test_text_file_metadata_link_cannot_be_a_script_url() -> None:
    for link in _UNSAFE_LINKS:
        documents = _process("notes.txt", _text_upload(link, "hash"), "text/plain")
        _assert_documents_are_safe(documents)


def test_html_comment_metadata_link_cannot_be_a_script_url() -> None:
    documents = _process(
        "notes.txt",
        _text_upload("javascript:void(0)", "html"),
        "text/plain",
    )
    _assert_documents_are_safe(documents)


def test_pdf_info_link_cannot_be_a_script_url() -> None:
    for link in ("javascript:void(0)", "java\tscript:void(0)", "data:text/html,hi"):
        documents = _process("notes.pdf", _pdf_upload(link), "application/pdf")
        _assert_documents_are_safe(documents)


def test_connector_metadata_link_cannot_be_a_script_url() -> None:
    file = BytesIO(f"{_BODY}\n".encode())
    documents = _process(
        "notes.txt",
        file,
        "text/plain",
        metadata={"link": "javascript:void(0)"},
    )
    _assert_documents_are_safe(documents)
