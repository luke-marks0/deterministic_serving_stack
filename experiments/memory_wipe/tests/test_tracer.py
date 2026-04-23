import time
from pose.tracer import Tracer


def test_tracer_collects_events():
    t = Tracer()
    with t.step("foo", bytes=1000):
        time.sleep(0.01)
    events = t.events()
    assert len(events) == 1
    assert events[0]["step"] == "foo"
    assert events[0]["bytes"] == 1000
    assert events[0]["end_ts"] > events[0]["start_ts"]


def test_tracer_multiple_events():
    t = Tracer()
    with t.step("a"):
        pass
    with t.step("b", bytes=500):
        pass
    events = t.events()
    assert len(events) == 2
    assert events[0]["step"] == "a"
    assert events[0]["bytes"] is None
    assert events[1]["step"] == "b"
    assert events[1]["bytes"] == 500


def test_tracer_events_are_ordered():
    t = Tracer()
    with t.step("first"):
        pass
    with t.step("second"):
        pass
    events = t.events()
    assert events[0]["start_ts"] <= events[1]["start_ts"]


def test_tracer_nested_is_flat():
    """Nested steps produce flat events, not a tree."""
    t = Tracer()
    with t.step("outer"):
        with t.step("inner"):
            pass
    events = t.events()
    assert len(events) == 2
    assert events[0]["step"] == "outer"
    assert events[1]["step"] == "inner"
