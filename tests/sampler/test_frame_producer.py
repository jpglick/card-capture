"""FrameProducer: order, first-frame gating, error propagation, early stop."""
import time

import numpy as np
import pytest

from card_capture.core.models import FrameSample
from card_capture.sampler.frame_producer import FrameProducer


def _frame(i: int) -> FrameSample:
    return FrameSample(
        frame_index=i,
        timestamp_ms=i * 33,
        image=np.zeros((2, 2, 3), np.uint8),
        width=2,
        height=2,
    )


class _FakeSampler:
    def __init__(self, n, *, boom_at=None, delay=0.0):
        self.n = n
        self.boom_at = boom_at
        self.delay = delay

    def sample(self):
        for i in range(self.n):
            if self.boom_at is not None and i == self.boom_at:
                raise RuntimeError("decode failed")
            if self.delay:
                time.sleep(self.delay)
            yield _frame(i)


def test_yields_all_frames_in_order():
    prod = FrameProducer(_FakeSampler(5)).start()
    assert [f.frame_index for f in prod] == [0, 1, 2, 3, 4]


def test_wait_first_unblocks_then_drains_all():
    prod = FrameProducer(_FakeSampler(3, delay=0.01)).start()
    assert prod.wait_first(timeout=2.0) is True
    assert [f.frame_index for f in prod] == [0, 1, 2]


def test_producer_error_propagates_to_consumer():
    prod = FrameProducer(_FakeSampler(5, boom_at=2)).start()
    with pytest.raises(RuntimeError, match="decode failed"):
        list(prod)


def test_stop_terminates_blocked_producer_without_hanging():
    prod = FrameProducer(_FakeSampler(10_000), maxsize=1).start()
    assert prod.wait_first(timeout=2.0) is True
    prod.stop(timeout=2.0)
    assert prod._thread is not None and not prod._thread.is_alive()
