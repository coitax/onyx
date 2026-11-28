import os

from onyx.utils.pillow_hardening import disable_jpeg2000_decoding

__version__ = os.environ.get("ONYX_VERSION", "") or "Development"

disable_jpeg2000_decoding()
