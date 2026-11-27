from typing import cast

import puremagic
from pydantic import BaseModel

from onyx.utils.logger import setup_logger

logger = setup_logger()

# Stored file types come from the uploader, so only types a browser will never run
# as an active document in our origin are served inline.
_INLINE_SAFE_MEDIA_TYPES = frozenset(
    {
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/gif",
        "image/webp",
        "image/bmp",
        "image/avif",
        "image/x-icon",
        "image/vnd.microsoft.icon",
        # SVG can carry script; it stays inline so <img> tags keep working, and the
        # sandboxing CSP below stops it from running when opened directly.
        "image/svg+xml",
        "application/pdf",
        "text/plain",
        "text/csv",
        "text/markdown",
    }
)

# Chrome's built-in PDF viewer does not render under a sandboxing CSP.
_NO_CSP_MEDIA_TYPES = frozenset({"application/pdf"})

STORED_FILE_CSP = (
    "default-src 'none'; img-src 'self' data:; style-src 'unsafe-inline'; sandbox"
)


def get_safe_file_response_params(
    stored_media_type: str | None,
) -> tuple[str, dict[str, str]]:
    """Returns the media type and headers to use when serving stored file bytes."""
    media_type = (stored_media_type or "").split(";", 1)[0].strip().lower()
    headers = {"X-Content-Type-Options": "nosniff"}

    if media_type not in _INLINE_SAFE_MEDIA_TYPES:
        media_type = "application/octet-stream"
        headers["Content-Disposition"] = "attachment"

    if media_type not in _NO_CSP_MEDIA_TYPES:
        headers["Content-Security-Policy"] = STORED_FILE_CSP

    return media_type, headers


class FileWithMimeType(BaseModel):
    data: bytes
    mime_type: str


class OnyxStaticFileManager:
    """Retrieve static resources with this class. Currently, these should all be located
    in the static directory ... e.g. static/images/logo.png"""

    @staticmethod
    def get_static(filename: str) -> FileWithMimeType | None:
        try:
            mime_type: str = "application/octet-stream"
            with open(filename, "rb") as f:
                file_content = f.read()
                matches = puremagic.magic_string(file_content)
                if matches:
                    mime_type = cast(str, matches[0].mime_type)
        except (OSError, FileNotFoundError, PermissionError) as e:
            logger.error(f"Failed to read file {filename}: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected exception reading file {filename}: {e}")
            return None

        return FileWithMimeType(data=file_content, mime_type=mime_type)
