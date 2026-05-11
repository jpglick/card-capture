"""Install benign-warning filters before heavy dependencies load.

Must be imported at the very top of `cli.py` so filters are in place before
`cv2`, `av`, or `urllib3` initialize."""
from __future__ import annotations

import warnings


def install() -> None:
    # urllib3 on macOS system Python links LibreSSL; harmless to us.
    warnings.filterwarnings(
        "ignore",
        message=r"urllib3 v2 only supports OpenSSL.*",
    )

    # huggingface_hub "unauthenticated requests" — we now reuse local cache (Task 3);
    # the warning still appears on first download. Hide the specific message.
    warnings.filterwarnings(
        "ignore",
        message=r".*sending unauthenticated requests to the HF Hub.*",
    )

    # Suppress objc duplicate-class console spam from cv2/.dylibs and av/.dylibs
    # shipping libavdevice on macOS. This spam bypasses Python's warning system,
    # so it cannot be filtered from Python. We document this as a known issue.
    pass
