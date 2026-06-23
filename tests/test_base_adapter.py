import pytest

from genailit import AdapterContext, BaseAgentAdapter, GenAILitEvent, TelemetryStore


def test_base_agent_adapter_is_abstract() -> None:
    with pytest.raises(TypeError):
        BaseAgentAdapter()  # type: ignore[abstract]


def test_base_agent_adapter_runs_and_records_telemetry() -> None:
    class DemoAdapter(BaseAgentAdapter):
        def build_events(self, context: AdapterContext):
            yield GenAILitEvent(name="ready", payload={"session_id": context.session_id})

    telemetry = TelemetryStore()
    context = AdapterContext(session_id="abc", telemetry=telemetry)

    events = DemoAdapter().run(context)

    assert events == (GenAILitEvent(name="ready", payload={"session_id": "abc"}),)
    assert telemetry.snapshot() == events


def test_base_agent_adapter_stream_delegates_to_sync_build_events() -> None:
    class DemoAdapter(BaseAgentAdapter):
        def build_events(self, context: AdapterContext):
            yield GenAILitEvent(name="ready", payload={"input": context.input_data})

    context = AdapterContext(session_id="abc", input_data={"prompt": "Hi"})

    events = tuple(DemoAdapter().stream(context.input_data, context))

    assert events == (
        GenAILitEvent(name="ready", payload={"input": {"prompt": "Hi"}}),
    )


def test_adapter_context_merges_metadata() -> None:
    context = AdapterContext(session_id="abc", metadata={"alpha": 1})

    updated = context.with_metadata(beta=2)

    assert updated.session_id == "abc"
    assert updated.metadata == {"alpha": 1, "beta": 2}
    assert context.metadata == {"alpha": 1}
