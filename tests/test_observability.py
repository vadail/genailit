from __future__ import annotations

import pytest

from genailit import GenAILitEvent, TelemetryStore


def test_telemetry_store_returns_session_trace_and_metrics(monkeypatch) -> None:
    store = TelemetryStore()
    times = iter([1.0, 1.2, 1.5, 1.8, 2.0, 2.3])
    monkeypatch.setattr("genailit.telemetry.time.perf_counter", lambda: next(times))

    session_id = "session-a"
    store.record(GenAILitEvent(name="session.started", payload={"input_tokens": 5}), session_id=session_id)
    store.record(GenAILitEvent(name="node.started", payload={"name": "demo"}), session_id=session_id)
    store.record(GenAILitEvent(name="agent.token", payload={"delta": "Hola mundo"}), session_id=session_id)
    store.record(GenAILitEvent(name="error", payload={"message": "boom"}), session_id=session_id)
    store.record(GenAILitEvent(name="retry.started"), session_id=session_id)
    store.record(GenAILitEvent(name="session.ended"), session_id=session_id)

    trace = store.get_session_trace(session_id)
    metrics = store.get_session_metrics(session_id)

    assert trace[0].name == "session.started"
    assert trace[-1].name == "session.ended"
    assert metrics["total_events"] == 6
    assert metrics["input_tokens"] == 5
    assert metrics["output_tokens"] == 2
    assert metrics["total_tokens"] == 7
    assert metrics["model"] is None
    assert metrics["provider"] is None
    assert metrics["latency_ms"] == pytest.approx(1300.0)
    assert metrics["ttft_ms"] == pytest.approx(500.0)
    assert metrics["cost_usd"] is None
    assert metrics["error_count"] == 1
    assert metrics["retry_count"] == 1
    assert metrics["estimated"] == {"tokens": True, "cost": True}
