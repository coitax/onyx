from typing import Any

from PIL import Image
from PIL import ImageFile
from PIL import Jpeg2KImagePlugin

# Every Pillow wheel from 11.1.0 through at least 13.0.0 bundles OpenJPEG
# 2.5.3/2.5.4, whose decoder has a heap buffer overflow write (OSV-2025-219)
# that is fixed upstream but not in any OpenJPEG release. Onyx never needs to
# read JPEG 2000, so untrusted codestreams must never reach that decoder.
_JPEG2000_FORMAT = Jpeg2KImagePlugin.Jpeg2KImageFile.format
_JPEG2000_DECODER = "jpeg2k"


class _RefusedJpeg2000Decoder(ImageFile.PyDecoder):
    def __init__(self, mode: str, *args: Any) -> None:  # noqa: ARG002
        raise OSError("JPEG 2000 decoding is disabled")


def disable_jpeg2000_decoding() -> None:
    """Make Pillow refuse to read JPEG 2000 anywhere in this process.

    Python decoders registered with Pillow take precedence over its bundled C
    decoders, so every load, ``Image.frombytes`` or direct use of the plugin
    class fails before OpenJPEG runs. Unregistering the opener additionally
    makes ``Image.open`` treat JPEG 2000 as an unidentified image. Encoding is
    left untouched.
    """
    Image.register_decoder(_JPEG2000_DECODER, _RefusedJpeg2000Decoder)
    Image.OPEN.pop(_JPEG2000_FORMAT, None)
    if _JPEG2000_FORMAT in Image.ID:
        Image.ID.remove(_JPEG2000_FORMAT)
