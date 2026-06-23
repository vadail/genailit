from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from genailit import AdapterContext, GenAILitEvent
from genailit.adapters.langgraph import LangGraphAdapter


class FakeMetadataGraph:
    def __init__(self) -> None:
        self.astream_called = False

    async def astream(
        self,
        input_data: object,
        *,
        input_key: str,
        stream_mode: str | list[str] | None,
        context: AdapterContext,
    ) -> AsyncIterator[dict[str, object]]:
        self.astream_called = True
        assert input_data == {"prompt": "Hola"}
        assert input_key == "messages"
        assert stream_mode is None
        assert context.session_id == "session-1"
        yield {
            "event": "on_chat_model_start",
            "metadata": {
                "ls_provider": "anthropic",
                "ls_model_name": "claude-3-haiku",
            },
        }
        yield {"event": "on_chat_model_stream", "token": "Hola "}
        yield {
            "event": "on_chat_model_end",
            "content": "Hola desde GenAILit",
            "response_metadata": {"model_name": "claude-3-sonnet"},
            "usage_metadata": {
                "input_tokens": 7,
                "output_tokens": 3,
                "total_tokens": 10,
            },
        }


class FakeTokenUsageGraph:
    def __init__(self) -> None:
        self.astream_called = False

    async def astream(
        self,
        input_data: object,
        *,
        input_key: str,
        stream_mode: str | list[str] | None,
        context: AdapterContext,
    ) -> AsyncIterator[dict[str, object]]:
        self.astream_called = True
        assert input_data == {"prompt": "Hola"}
        assert input_key == "messages"
        assert stream_mode is None
        assert context.session_id == "session-2"
        yield {
            "event": "on_llm_end",
            "llm_output": {
                "token_usage": {
                    "prompt_tokens": 4,
                    "completion_tokens": 6,
                    "total_tokens": 10,
                }
            },
        }


class FakeLegacyGraph:
    def __init__(self) -> None:
        self.astream_called = False
        self.astream_events_called = False

    async def astream_events(
        self,
        input_data: object,
        *,
        input_key: str,
        stream_mode: str | list[str] | None,
        context: AdapterContext,
    ) -> AsyncIterator[dict[str, object]]:
        self.astream_events_called = True
        assert input_data == {"prompt": "Hola"}
        assert input_key == "messages"
        assert stream_mode == "updates"
        assert context.include_raw is True
        yield {"event": "on_chain_start", "name": "legacy"}
        yield {"event": "on_llm_stream", "token": "Legacy"}
        yield {"event": "on_chain_end", "name": "legacy"}


class FakeFinalOutputGraph:
    def __init__(self) -> None:
        self.astream_called = False

    async def astream(
        self,
        input_data: object,
        *,
        input_key: str,
        stream_mode: str | list[str] | None,
        context: AdapterContext,
    ) -> AsyncIterator[dict[str, object]]:
        self.astream_called = True
        assert input_data == {"prompt": "Hola"}
        assert input_key == "messages"
        assert stream_mode is None
        assert context.session_id == "session-3"
        yield {"messages": ["Hello", "hola"]}


class FakeNestedChunkGraph:
    def __init__(self) -> None:
        self.astream_called = False

    async def astream(
        self,
        input_data: object,
        *,
        input_key: str,
        stream_mode: str | list[str] | None,
        context: AdapterContext,
    ) -> AsyncIterator[dict[str, object]]:
        self.astream_called = True
        assert input_data == {"messages": ["Hello"]}
        assert input_key == "messages"
        assert stream_mode is None
        assert context.session_id == "session-4"
        yield {"demo_node": {"messages": ["Hello", "Hola desde LangGraph + GenAILit"]}}


def _collect(stream: AsyncIterator[GenAILitEvent]) -> list[GenAILitEvent]:
    async def _run() -> list[GenAILitEvent]:
        return [event async for event in stream]

    return asyncio.run(_run())


def _payloads(events: list[GenAILitEvent], name: str) -> list[dict[str, object]]:
    return [event.payload for event in events if event.name == name]


def test_langgraph_adapter_emits_metadata_and_metrics_from_usage_metadata() -> None:
    graph = FakeMetadataGraph()
    adapter = LangGraphAdapter(graph=graph)
    context = AdapterContext(session_id="session-1", run_id="run-1")

    events = _collect(adapter.stream({"prompt": "Hola"}, context))

    assert graph.astream_called is True
    assert _payloads(events, "llm.started") == [
        {
            "provider": "anthropic",
            "model": "claude-3-haiku",
            "metadata": {"source": "langgraph"},
            "timing": {"latency_ms": None, "ttft_ms": None},
        }
    ]
    assert {payload["delta"] for payload in _payloads(events, "agent.token")} == {
        "Hola ",
        "Hola desde GenAILit",
    }
    assert _payloads(events, "agent.message") == [{"content": "Hola desde GenAILit"}]
    llm_ended_payload = _payloads(events, "llm.ended")[0]
    assert llm_ended_payload["provider"] == "anthropic"
    assert llm_ended_payload["model"] == "claude-3-sonnet"
    assert llm_ended_payload["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert llm_ended_payload["metadata"] == {"source": "langgraph"}
    assert llm_ended_payload["timing"]["latency_ms"] is not None
    assert llm_ended_payload["timing"]["ttft_ms"] is not None
    metrics_payload = _payloads(events, "metrics.updated")[0]
    assert metrics_payload["metadata"] == {"source": "langgraph"}
    assert metrics_payload["usage"] == {
        "input_tokens": 7,
        "output_tokens": 3,
        "total_tokens": 10,
    }
    assert metrics_payload["provider"] == "anthropic"
    assert metrics_payload["model"] == "claude-3-sonnet"
    assert metrics_payload["timing"]["latency_ms"] is not None
    assert metrics_payload["timing"]["ttft_ms"] is not None
    assert all("raw" not in payload for event in events for payload in [event.payload])


def test_langgraph_adapter_converts_llm_output_token_usage() -> None:
    graph = FakeTokenUsageGraph()
    adapter = LangGraphAdapter(graph=graph)
    context = AdapterContext(session_id="session-2", run_id="run-2")

    events = _collect(adapter.stream({"prompt": "Hola"}, context))

    assert graph.astream_called is True
    assert _payloads(events, "llm.ended")[0]["usage"] == {
        "input_tokens": 4,
        "output_tokens": 6,
        "total_tokens": 10,
    }
    assert _payloads(events, "metrics.updated")[0]["usage"] == {
        "input_tokens": 4,
        "output_tokens": 6,
        "total_tokens": 10,
    }


def test_langgraph_adapter_uses_legacy_fallback_and_includes_raw() -> None:
    graph = FakeLegacyGraph()
    adapter = LangGraphAdapter(graph=graph, stream_mode="updates")
    context = AdapterContext(session_id="session-5", run_id="run-5", include_raw=True)

    events = _collect(adapter.stream({"prompt": "Hola"}, context))

    assert graph.astream_called is False
    assert graph.astream_events_called is True
    assert any(
        event.name == "node.started" and event.payload.get("raw") == {"event": "on_chain_start", "name": "legacy"}
        for event in events
    )
    assert any(
        event.name == "agent.token" and event.payload.get("raw") == {"event": "on_llm_stream", "token": "Legacy"}
        for event in events
    )
    assert any(
        event.name == "node.ended" and event.payload.get("raw") == {"event": "on_chain_end", "name": "legacy"}
        for event in events
    )


def test_langgraph_adapter_raises_clear_error_without_stream_method() -> None:
    adapter = LangGraphAdapter(graph=object())

    events = _collect(adapter.stream({}, AdapterContext(session_id="session-6")))

    assert events[2].name == "error"
    assert "genailit[langgraph]" in events[2].payload["message"]


def test_langgraph_adapter_converts_final_messages_to_tokens() -> None:
    graph = FakeFinalOutputGraph()
    adapter = LangGraphAdapter(graph=graph)
    context = AdapterContext(session_id="session-3", run_id="run-3")

    events = _collect(adapter.stream({"prompt": "Hola"}, context))

    assert graph.astream_called is True
    assert any(event.name == "agent.token" and event.payload == {"delta": "hola"} for event in events)
    assert any(event.name == "agent.message" and event.payload == {"content": "hola"} for event in events)


def test_langgraph_adapter_extracts_text_from_nested_node_chunks() -> None:
    graph = FakeNestedChunkGraph()
    adapter = LangGraphAdapter(graph=graph)
    context = AdapterContext(session_id="session-4", run_id="run-4")

    events = _collect(adapter.stream({"messages": ["Hello"]}, context))

    assert graph.astream_called is True
    assert any(
        event.name == "agent.token"
        and event.payload == {"delta": "Hola desde LangGraph + GenAILit"}
        for event in events
    )
