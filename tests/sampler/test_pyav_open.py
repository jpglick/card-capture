"""_open_pyav_container must use plain software decode (no hwaccel option)."""
from pathlib import Path
from unittest.mock import MagicMock, patch

from card_capture.sampler import VideoSampler


def test_open_pyav_container_uses_software_no_hwaccel():
    with patch("av.open") as mock_open:
        mock_open.return_value = MagicMock(name="container")
        container, hw = VideoSampler._open_pyav_container(Path("/tmp/x.mov"))

    assert hw is False
    assert mock_open.call_count == 1
    args, kwargs = mock_open.call_args
    assert "options" not in kwargs
    assert args == ("/tmp/x.mov",)
    assert container is mock_open.return_value
