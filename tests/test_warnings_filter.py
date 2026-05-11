import subprocess
import sys


def test_cli_invocation_has_no_objc_or_urllib3_warnings():
    """A `card-capture --help` invocation must not emit the known-benign macOS/urllib3 warnings."""
    result = subprocess.run(
        [sys.executable, "-m", "card_capture.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    combined = result.stdout + result.stderr
    assert "AVFFrameReceiver" not in combined, combined
    assert "NotOpenSSLWarning" not in combined, combined
