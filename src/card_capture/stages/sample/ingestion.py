from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol

from card_capture.core.models import FramePacket
from card_capture.core.video_utils import (
    FrameTriageFilter,
    RollingWindowTriage,
    _decord_available,
    _open_capture,
    _resolve_reader_backend,
    probe_video,
)


class FrameReader(Protocol):
    def iter_frames(self, video_path: Path | str) -> Iterator[FramePacket]:
        ...
