from card_capture.models import TrackState


def test_trackstate_session_id_defaults_zero():
    ts = TrackState(instance_id="i")
    assert ts.session_id == 0


def test_trackstate_session_id_settable():
    ts = TrackState(instance_id="i", session_id=3)
    assert ts.session_id == 3

