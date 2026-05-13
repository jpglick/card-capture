"""Apple Silicon hardware acceleration detection and initialization.

Provides feature detection for ANE (Neural Engine), VideoToolbox (Hardware
Decoder), and vImage (Accelerate framework).
"""
from __future__ import annotations

import platform
import subprocess
from typing import Dict, Any


def get_apple_silicon_capabilities() -> Dict[str, Any]:
    """Detect specific hardware capabilities on macOS."""
    capabilities = {
        "is_macos": platform.system() == "Darwin",
        "is_apple_silicon": False,
        "has_ane": False,
        "has_videotoolbox": False,
        "has_vimage": False,
        "chip_name": "unknown",
    }

    if not capabilities["is_macos"]:
        return capabilities

    # 1. Check for Apple Silicon (M1, M2, etc.)
    try:
        brand = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        capabilities["chip_name"] = brand
        if "Apple" in brand:
            capabilities["is_apple_silicon"] = True
            capabilities["has_ane"] = True # All Apple Silicon has ANE
    except Exception:
        pass

    # 2. VideoToolbox detection
    # On macOS, VideoToolbox is always present if system is recent enough
    capabilities["has_videotoolbox"] = True

    # 3. vImage / Accelerate detection
    # Present on all macOS
    capabilities["has_vimage"] = True

    return capabilities


def print_capabilities():
    caps = get_apple_silicon_capabilities()
    print("Apple Silicon Capabilities:")
    for k, v in caps.items():
        print(f"  {k:20}: {v}")


if __name__ == "__main__":
    print_capabilities()
