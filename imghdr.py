"""
Compatibility shim for the deprecated stdlib module ``imghdr``.

Python 3.13 removed :mod:`imghdr`, but some third‑party libraries (e.g.
PaddleOCR 2.x) still import it. In this codebase those libraries only
need the module to exist; they do not rely on its functionality.

We therefore provide a minimal stub that satisfies ``import imghdr`` and
exposes a ``what`` function with the same signature. It always returns
``None``, which is sufficient for our current usage.
"""

from __future__ import annotations
from typing import Optional, Union, BinaryIO
import io


def what(file: Union[str, bytes, BinaryIO], h: Optional[bytes] = None) -> Optional[str]:
    """
    Minimal replacement for :func:`imghdr.what`.

    Parameters
    ----------
    file:
        Filename, path-like, bytes buffer, or file-like object.
    h:
        Optional bytes containing the image header. Ignored here.

    Returns
    -------
    str | None
        Always returns ``None`` in this lightweight shim.
    """

    # Accept the same kinds of inputs as the original API but deliberately
    # do not attempt to sniff image types. Returning None is treated as
    # "unknown" by callers.
    if isinstance(file, (str, bytes)):
        # If a path or bytes are passed we could try to open/read, but
        # for our current dependencies this is unnecessary.
        return None

    if isinstance(file, io.IOBase):
        return None

    return None