from smoke_monitor.alarm import AlarmStateMachine


def test_alarm_requires_consecutive_positive_predictions():
    machine = AlarmStateMachine(
        positive_frames_to_alarm=3,
        negative_frames_to_clear=2,
        alert_repeat_seconds=100,
    )
    assert machine.update(False, 0) == []
    assert machine.update(True, 1) == []
    assert machine.update(False, 2) == []
    assert machine.update(True, 3) == []
    assert machine.update(True, 4) == []
    events = machine.update(True, 5)
    assert [event.event_type for event in events] == ["smoke_started"]
    assert machine.state == "ALARM"


def test_alarm_clears_after_consecutive_negative_predictions():
    machine = AlarmStateMachine(positive_frames_to_alarm=1, negative_frames_to_clear=2)
    assert [event.event_type for event in machine.update(True, 0)] == ["smoke_started"]
    assert machine.update(False, 1) == []
    assert [event.event_type for event in machine.update(False, 2)] == ["smoke_cleared"]
    assert machine.state == "NORMAL"


def test_persistent_smoke_can_send_a_reminder_after_cooldown():
    machine = AlarmStateMachine(
        positive_frames_to_alarm=1,
        negative_frames_to_clear=2,
        alert_repeat_seconds=10,
    )
    assert [event.event_type for event in machine.update(True, 0)] == ["smoke_started"]
    assert machine.update(True, 5) == []
    assert [event.event_type for event in machine.update(True, 10)] == ["smoke_reminder"]

