from __future__ import annotations

import pytest

from genailit import GenAILitEvent, TelemetryStore


def test_telemetry_store_records_and_clears_events() -> None:
    store = TelemetryStore()
    first = GenAILitEvent(name="a")
    second = GenAILitEvent(name="b")

    store.record(first)
    store.extend([second])

    assert store.snapshot() == (first, second)

    store.clear()

    assert store.snapshot() == ()


def test_telemetry_metrics_use_v2_usage_payload(monkeypatch) -> None:
    store = TelemetryStore()
    times = iter([1.0, 1.1, 1.2, 1.3])
    monkeypatch.setattr("genailit.telemetry.time.perf_counter", lambda: next(times))

    session_id = "session-v2"
    store.record(
        GenAILitEvent(
            name="session.started",
            payload={
                "usage": {
                    "input_tokens": 12,
                    "output_tokens": 34,
                    "total_tokens": 46,
                },
                "provider": "openai",
                "model": "gpt-4.1-mini",
                "timing": {
                    "latency_ms": 987.6,
                    "ttft_ms": 120.4,
                },
                "cost": {"cost_usd": 0.0123},
            },
        ),
        session_id=session_id,
    )
    store.record(GenAILitEvent(name="retry.started"), session_id=session_id)
    store.record(GenAILitEvent(name="error", payload={"message": "boom"}), session_id=session_id)
    store.record(GenAILitEvent(name="session.ended"), session_id=session_id)

    metrics = store.get_session_metrics(session_id)

    assert metrics == {
        "total_events": 4,
        "input_tokens": 12,
        "output_tokens": 34,
        "total_tokens": 46,
        "model": "gpt-4.1-mini",
        "provider": "openai",
        "latency_ms": 987.6,
        "ttft_ms": 120.4,
        "cost_usd": 0.0123,
        "error_count": 1,
        "retry_count": 1,
        "estimated": {"tokens": False, "cost": False},
    }


def test_telemetry_metrics_support_legacy_aliases_metadata_cost_and_counts() -> None:
    store = TelemetryStore()

    session_id = "session-legacy"
    store.record(
        GenAILitEvent(
            name="session.started",
            payload={
                "prompt_tokens": 7,
                "completion_tokens": 3,
                "metadata": {
                    "ls_provider": "anthropic",
                    "ls_model_name": "claude-3-haiku",
                },
                "timing": {
                    "latency_ms": 250.0,
                    "ttft_ms": 80.0,
                },
                "cost_usd": 0.25,
            },
        ),
        session_id=session_id,
    )
    store.record(GenAILitEvent(name="retry.started", payload={"retry_count": 2}), session_id=session_id)
    store.record(
        GenAILitEvent(name="error", payload={"message": "boom"}),
        session_id=session_id,
    )
    store.record(
        GenAILitEvent(name="session.ended", payload={"response_metadata": {"model_name": "claude-3-sonnet"}}),
        session_id=session_id,
    )

    metrics = store.get_session_metrics(session_id)

    assert metrics["input_tokens"] == 7
    assert metrics["output_tokens"] == 3
    assert metrics["total_tokens"] == 10
    assert metrics["provider"] == "anthropic"
    assert metrics["model"] == "claude-3-sonnet"
    assert metrics["latency_ms"] == 250.0
    assert metrics["ttft_ms"] == 80.0
    assert metrics["cost_usd"] == 0.25
    assert metrics["error_count"] == 1
    assert metrics["retry_count"] == 2
    assert metrics["estimated"] == {"tokens": False, "cost": False}


def test_telemetry_metrics_estimate_tokens_and_timings_from_events(monkeypatch) -> None:
    store = TelemetryStore()
    times = iter([10.0, 10.4, 10.7, 11.3])
    monkeypatch.setattr("genailit.telemetry.time.perf_counter", lambda: next(times))

    session_id = "session-estimated"
    store.record(GenAILitEvent(name="session.started"), session_id=session_id)
    store.record(GenAILitEvent(name="node.started", payload={"name": "demo"}), session_id=session_id)
    store.record(GenAILitEvent(name="agent.token", payload={"delta": "Hola mundo"}), session_id=session_id)
    store.record(GenAILitEvent(name="session.ended"), session_id=session_id)

    metrics = store.get_session_metrics(session_id)

    assert metrics["input_tokens"] == 0
    assert metrics["output_tokens"] == 2
    assert metrics["total_tokens"] == 2
    assert metrics["latency_ms"] == pytest.approx(1300.0)
    assert metrics["ttft_ms"] == pytest.approx(700.0)
    assert metrics["estimated"] == {"tokens": True, "cost": True}


def test_telemetry_interleaved_sessions_do_not_leak_metrics(monkeypatch) -> None:
    store = TelemetryStore()
    times = iter([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    monkeypatch.setattr("genailit.telemetry.time.perf_counter", lambda: next(times))

    store.record(GenAILitEvent(name="session.started"), session_id="session-a")
    store.record(
        GenAILitEvent(
            name="metrics.updated",
            payload={
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 20,
                    "total_tokens": 30,
                },
                "provider": "openai",
                "model": "gpt-a",
            },
        ),
        session_id="session-b",
    )
    store.record(GenAILitEvent(name="agent.token", payload={"delta": "Hola A"}), session_id="session-a")
    store.record(GenAILitEvent(name="error", payload={"message": "boom"}), session_id="session-b")
    store.record(GenAILitEvent(name="session.ended"), session_id="session-a")
    store.record(GenAILitEvent(name="session.ended"), session_id="session-b")

    metrics_a = store.get_session_metrics("session-a")
    metrics_b = store.get_session_metrics("session-b")

    assert metrics_a["total_events"] == 3
    assert metrics_a["output_tokens"] == 2
    assert metrics_a["provider"] is None
    assert metrics_a["error_count"] == 0
    assert metrics_a["estimated"] == {"tokens": True, "cost": True}

    assert metrics_b["total_events"] == 3
    assert metrics_b["input_tokens"] == 10
    assert metrics_b["output_tokens"] == 20
    assert metrics_b["total_tokens"] == 30
    assert metrics_b["provider"] == "openai"
    assert metrics_b["model"] == "gpt-a"
    assert metrics_b["error_count"] == 1
    assert metrics_b["estimated"] == {"tokens": False, "cost": True}
