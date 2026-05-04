from __future__ import annotations

import platform
import subprocess
import sys


def main() -> int:
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"machine: {platform.machine()}")

    try:
        sw_vers = subprocess.run(
            ["sw_vers"],
            capture_output=True,
            text=True,
            check=False,
        )
        if sw_vers.stdout.strip():
            print("sw_vers:")
            print(sw_vers.stdout.strip())
    except Exception as exc:
        print(f"sw_vers_error: {exc!r}")

    try:
        import torch
    except Exception as exc:
        print(f"torch_import_error: {exc!r}")
        return 1

    print(f"torch: {torch.__version__}")
    print(f"mps_built: {torch.backends.mps.is_built()}")
    print(f"mps_available: {torch.backends.mps.is_available()}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    try:
        cpu_tensor = torch.ones(1)
        print(f"cpu_tensor: {cpu_tensor}")
    except Exception as exc:
        print(f"cpu_tensor_error: {exc!r}")

    try:
        mps_tensor = torch.ones(1, device="mps")
        print(f"mps_tensor: {mps_tensor}")
    except Exception as exc:
        print(f"mps_tensor_error: {exc!r}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
