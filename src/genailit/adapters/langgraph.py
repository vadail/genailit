from __future__ import annotations

import time
from collections.abc import AsyncIterable, AsyncIterator, Mapping
from dataclasses import asdict, dataclass, is_dataclass
from typing import Any
from uuid import uuid4

from .base import AdapterContext, BaseAgentAdapter
from ..events import GenAILitEvent
from ..introspection import introspect_system
from ..pricing import estimate_cost_usd


_CLEAR_ERROR = (
    'LangGraphAdapter requires a graph with "astream" or "astream_events". '
    'Install it with pip install "genailit[langgraph]".'
)

_STREAM_EVENTS = {"agent.token"}
_LLM_START_EVENTS = {"llm.started"}
_LLM_END_EVENTS = {"llm.ended"}
_NODE_START_EVENTS = {"node.started"}
_NODE_END_EVENTS = {"node.ended"}
_TOOL_START_EVENTS = {"tool.started"}
_TOOL_END_EVENTS = {"tool.ended"}


@dataclass(slots=True)
class _TranslationResult:
    events: list[GenAILitEvent]
    visible_output: bool = False
    provider: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None


@dataclass(slots=True)
class LangGraphAdapter(BaseAgentAdapter):
    graph: Any
    input_key: str = "messages"
    stream_mode: str | list[str] | None = None
    include_raw: bool = False

    def get_system_manifest(self) -> dict[str, Any]:
        return introspect_system(self.graph)

    async def stream(
        self,
        input_data: Any,
        context: AdapterContext | None = None,
    ) -> AsyncIterator[GenAILitEvent]:
        runtime_context = self._resolve_context(input_data, context)
        want_raw = self.include_raw or runtime_context.include_raw
        started_at = time.perf_counter()
        first_output_at: float | None = None
        last_provider: str | None = None
        last_model: str | None = None

        yield GenAILitEvent(
            name="session.started",
            payload={
                "session_id": runtime_context.session_id,
                "run_id": runtime_context.run_id,
            },
        )
        yield GenAILitEvent(
            name="node.started",
            payload={
                "session_id": runtime_context.session_id,
                "run_id": runtime_context.run_id,
            },
        )

        emitted = 0

        try:
            async for item in self._iterate_graph(input_data, runtime_context):
                now = time.perf_counter()
                translation = self._translate_item(
                    item,
                    want_raw,
                    started_at=started_at,
                    first_output_at=first_output_at,
                    now=now,
                    provider=last_provider,
                    model=last_model,
                )
                for event in translation.events:
                    emitted += 1
                    yield event
                if translation.provider is not None:
                    last_provider = translation.provider
                if translation.model is not None:
                    last_model = translation.model
                if translation.visible_output and first_output_at is None:
                    first_output_at = now
        except Exception as exc:
            yield self._event(
                "error",
                {
                    "message": str(exc),
                    "type": exc.__class__.__name__,
                },
            )
        finally:
            yield self._event("node.ended", {"run_id": runtime_context.run_id})
            yield self._event(
                "session.ended",
                {
                    "session_id": runtime_context.session_id,
                    "run_id": runtime_context.run_id,
                },
            )

    def _resolve_context(
        self,
        input_data: Any,
        context: AdapterContext | None,
    ) -> AdapterContext:
        if context is not None:
            return context
        return AdapterContext(
            session_id=uuid4().hex,
            run_id=uuid4().hex,
            input_data=input_data,
            include_raw=self.include_raw,
        )

    async def _iterate_graph(
        self,
        input_data: Any,
        context: AdapterContext,
    ) -> AsyncIterator[Any]:
        graph = self.graph

        from langchain_core.messages import HumanMessage

        graph_input = input_data

        if (
            isinstance(input_data, Mapping)
            and "prompt" in input_data
            and "messages" not in input_data
        ):
            graph_input = {
                "messages": [
                    HumanMessage(content=str(input_data["prompt"]))
                ]
            }

        config = {
            "configurable": {
                "thread_id": context.session_id,
            }
        }

        if hasattr(graph, "astream"):
            stream = graph.astream(
                graph_input,
                config=config,
                stream_mode=self.stream_mode,
            )

        elif hasattr(graph, "astream_events"):
            stream = graph.astream_events(
                input_data,
                config=config,
                stream_mode=self.stream_mode,
            )

        else:
            raise RuntimeError(_CLEAR_ERROR)

        async for item in stream:
            yield item
     
    def _translate_item(
        self,
        item: Any,
        want_raw: bool,
        *,
        started_at: float,
        first_output_at: float | None,
        now: float,
        provider: str | None,
        model: str | None,
    ) -> _TranslationResult:
        print("\n===== TRANSLATE ITEM =====")
        print(type(item))
        print(item)
        normalized = self._normalize_item(item, provider=provider, model=model)
        print("\n===== NORMALIZED =====")
        print(normalized)
        event_name = self._event_name(item)
        raw = self._serializable(item) if want_raw else None

        if event_name in _LLM_START_EVENTS:
            payload = self._llm_payload(
                normalized,
                stage="started",
                started_at=started_at,
                first_output_at=first_output_at,
                now=now,
            )
            if raw is not None:
                payload["raw"] = raw
            return _TranslationResult(
                [GenAILitEvent(name="llm.started", payload=payload)],
                provider=normalized["provider"],
                model=normalized["model"],
                usage=normalized["usage"],
            )

        if event_name in _LLM_END_EVENTS:
            events: list[GenAILitEvent] = []
            final_text = self._extract_final_text(item)
            if final_text is not None:
                token_payload = {"delta": final_text}
                message_payload = {"content": final_text}
                if raw is not None:
                    token_payload["raw"] = raw
                    message_payload["raw"] = raw
                events.append(GenAILitEvent(name="agent.token", payload=token_payload))
                events.append(GenAILitEvent(name="agent.message", payload=message_payload))

            payload = self._llm_payload(
                normalized,
                stage="ended",
                started_at=started_at,
                first_output_at=first_output_at,
                now=now,
            )
            if raw is not None:
                payload["raw"] = raw
            events.append(GenAILitEvent(name="llm.ended", payload=payload))
            if normalized["usage"] is not None:
                metrics_payload = self._metrics_payload(normalized, started_at, first_output_at, now)
                if raw is not None:
                    metrics_payload["raw"] = raw
                events.append(GenAILitEvent(name="metrics.updated", payload=metrics_payload))
            return _TranslationResult(
                events,
                visible_output=final_text is not None,
                provider=normalized["provider"],
                model=normalized["model"],
                usage=normalized["usage"],
            )

        if event_name in _STREAM_EVENTS:
            text = self._extract_stream_text(item)
            if text is None:
                return _TranslationResult([])
            payload = {"delta": text}
            if raw is not None:
                payload["raw"] = raw
            return _TranslationResult(
                [GenAILitEvent(name="agent.token", payload=payload)],
                visible_output=True,
                provider=normalized["provider"],
                model=normalized["model"],
                usage=normalized["usage"],
            )

        if event_name in _NODE_START_EVENTS | _NODE_END_EVENTS | _TOOL_START_EVENTS | _TOOL_END_EVENTS:
            payload = self._event_payload(item, event_name, explicit_event=True)
            if raw is not None:
                payload["raw"] = raw
            return _TranslationResult([GenAILitEvent(name=event_name, payload=payload)])

        if event_name == "metrics.updated":
            if normalized["usage"] is None:
                return _TranslationResult([])
            payload = self._metrics_payload(normalized, started_at, first_output_at, now)
            if raw is not None:
                payload["raw"] = raw
            return _TranslationResult(
                [GenAILitEvent(name="metrics.updated", payload=payload)],
                provider=normalized["provider"],
                model=normalized["model"],
                usage=normalized["usage"],
            )

        if event_name == "error":
            payload = self._event_payload(item, "error", explicit_event=True)
            if raw is not None:
                payload["raw"] = raw
            return _TranslationResult([GenAILitEvent(name="error", payload=payload)])

        events: list[GenAILitEvent] = []
        final_text = self._extract_final_text(item)
        if final_text is not None:
            token_payload = {"delta": final_text}
            message_payload = {"content": final_text}
            if raw is not None:
                token_payload["raw"] = raw
                message_payload["raw"] = raw
            events.append(GenAILitEvent(name="agent.token", payload=token_payload))
            events.append(GenAILitEvent(name="agent.message", payload=message_payload))
        if normalized["usage"] is not None:
            payload = self._metrics_payload(normalized, started_at, first_output_at, now)
            if raw is not None:
                payload["raw"] = raw
            events.append(GenAILitEvent(name="metrics.updated", payload=payload))
        return _TranslationResult(
            events,
            visible_output=final_text is not None,
            provider=normalized["provider"],
            model=normalized["model"],
            usage=normalized["usage"],
        )

    def _event_name(self, item: Any) -> str | None:
        if isinstance(item, Mapping):
            raw = item.get("event") or item.get("type") or item.get("kind")
            if raw is None:
                return None
            return self._map_event_name(str(raw))
        if isinstance(item, str):
            return "agent.token"
        return None

    def _map_event_name(self, raw: str) -> str | None:
        normalized = raw.lower().replace("-", "_")
        mapping = {
            "node.started": "node.started",
            "node_started": "node.started",
            "on_chain_start": "node.started",
            "on_graph_start": "node.started",
            "node.ended": "node.ended",
            "node_ended": "node.ended",
            "on_chain_end": "node.ended",
            "on_graph_end": "node.ended",
            "tool.started": "tool.started",
            "tool_started": "tool.started",
            "on_tool_start": "tool.started",
            "tool.ended": "tool.ended",
            "tool_ended": "tool.ended",
            "on_tool_end": "tool.ended",
            "llm.started": "llm.started",
            "on_llm_start": "llm.started",
            "on_chat_model_start": "llm.started",
            "llm.ended": "llm.ended",
            "on_llm_end": "llm.ended",
            "on_chat_model_end": "llm.ended",
            "agent.token": "agent.token",
            "token": "agent.token",
            "on_llm_stream": "agent.token",
            "on_chat_model_stream": "agent.token",
            "agent.message": "agent.message",
            "message": "agent.message",
            "metrics.updated": "metrics.updated",
            "metrics_updated": "metrics.updated",
            "error": "error",
        }
        return mapping.get(normalized)

    def _event_payload(
        self,
        item: Any,
        event_name: str,
        explicit_event: bool,
    ) -> dict[str, Any]:
        if isinstance(item, Mapping):
            if "token" in item:
                return {"delta": item["token"]}
            if "content" in item:
                return {"content": item["content"]}
            if "message" in item:
                return {"message": self._serializable(item["message"])}
            if self.input_key in item:
                return {"message": self._serializable(item[self.input_key])}
            if "name" in item:
                return {"name": item["name"]}
        if isinstance(item, str):
            return {"token": item}
        return {"message": self._serializable(item)}

    def _final_texts(self, item: Any) -> list[str]:
        texts: list[str] = []
        self._collect_final_texts(item, texts)
        return texts

    def _collect_final_texts(self, value: Any, texts: list[str]) -> None:
        if isinstance(value, Mapping):
            direct_text = self._direct_text(value)
            if direct_text is not None:
                texts.append(direct_text)
                return

            for child in value.values():
                self._collect_final_texts(child, texts)
            return

        text = self._text_from_value(value)
        if text is not None:
            texts.append(text)

    def _text_from_messages(self, value: Any) -> str | None:
        if isinstance(value, list):
            parts = [text for item in value if (text := self._text_from_value(item))]
            return parts[-1] if parts else None
        return self._text_from_value(value)

    def _direct_text(self, item: Mapping[str, Any]) -> str | None:
        if self.input_key in item:
            return self._text_from_messages(item[self.input_key])

        for key in ("content", "output", "response"):
            if key in item:
                return self._text_from_value(item[key])

        return None

    def _text_from_value(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in ("content", "output", "response", "text", "message"):
                if key in value:
                    text = self._text_from_value(value[key])
                    if text is not None:
                        return text
            if "messages" in value:
                return self._text_from_messages(value["messages"])
            return None
        if isinstance(value, list):
            parts = [text for item in value if (text := self._text_from_value(item))]
            return parts[-1] if parts else None
        if isinstance(value, tuple):
            parts = [text for item in value if (text := self._text_from_value(item))]
            return parts[-1] if parts else None
        if hasattr(value, "content"):
            content = getattr(value, "content")
            if isinstance(content, str):
                return content
        if hasattr(value, "model_dump"):
            return self._text_from_value(value.model_dump())
        if hasattr(value, "dict"):
            return self._text_from_value(value.dict())
        return None

    def _extract_stream_text(self, item: Any) -> str | None:
        if isinstance(item, Mapping):
            for key in ("token", "delta", "content", "text"):
                value = item.get(key)
                if isinstance(value, str) and value:
                    return value
            data = item.get("data")
            if data is not None:
                return self._extract_stream_text(data)
        return self._extract_text_from_value(item)

    def _extract_final_text(self, item: Any) -> str | None:
        text = self._final_text(item)
        if text is not None:
            return text
        return self._extract_stream_text(item)

    def _final_text(self, item: Any) -> str | None:
        texts: list[str] = []
        self._collect_final_texts(item, texts)
        if texts:
            return texts[-1]
        return None

    def _normalize_item(
        self,
        item: Any,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:

        extracted_provider = self._extract_text_field(
            item,
            (
                ("provider",),
                ("metadata", "ls_provider"),
                ("metadata", "provider"),
                ("response_metadata", "provider"),
                ("response_metadata", "model_provider"),
            ),
        )

        extracted_model = self._extract_text_field(
            item,
            (
                ("model",),
                ("metadata", "ls_model_name"),
                ("response_metadata", "model_name"),
                ("response_metadata", "model"),
            ),
        )

        usage = self._extract_usage(item)

        return {
            "provider": (
                extracted_provider
                if extracted_provider is not None
                else provider
            ),
            "model": (
                extracted_model
                if extracted_model is not None
                else model
            ),
            "usage": usage,
            "metadata": {"source": "langgraph"},
        }

    def _llm_payload(
        self,
        normalized: dict[str, Any],
        *,
        stage: str,
        started_at: float,
        first_output_at: float | None,
        now: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "metadata": {"source": "langgraph"},
            "timing": self._timing_payload(stage, started_at, first_output_at, now),
        }
        if normalized["provider"] is not None:
            payload["provider"] = normalized["provider"]
        if normalized["model"] is not None:
            payload["model"] = normalized["model"]
        if stage == "ended" and normalized["usage"] is not None:
            payload["usage"] = normalized["usage"]
        return payload

    def _metrics_payload(
        self,
        normalized: dict[str, Any],
        started_at: float,
        first_output_at: float | None,
        now: float,
    ) -> dict[str, Any]:

        payload: dict[str, Any] = {
            "metadata": {"source": "langgraph"},
            "timing": self._timing_payload(
                "ended",
                started_at,
                first_output_at,
                now,
            ),
        }

        if normalized["provider"] is not None:
            payload["provider"] = normalized["provider"]

        if normalized["model"] is not None:
            payload["model"] = normalized["model"]

        usage = normalized.get("usage")

        if usage is not None:
            payload["usage"] = usage

        cost_usd = None

        if (
            usage is not None
            and normalized["provider"] is not None
            and normalized["model"] is not None
        ):
            cost_usd = estimate_cost_usd(
                normalized["provider"],
                normalized["model"],
                usage.get("input_tokens", 0),
                usage.get("output_tokens", 0),
            )

        payload["cost"] = {
            "cost_usd": cost_usd
        }

        return payload

    def _timing_payload(
        self,
        stage: str,
        started_at: float,
        first_output_at: float | None,
        now: float,
    ) -> dict[str, float | None]:
        if stage == "started":
            return {"latency_ms": None, "ttft_ms": None}
        latency_ms = (now - started_at) * 1000.0
        ttft_ms = None
        if first_output_at is not None:
            ttft_ms = (first_output_at - started_at) * 1000.0
        return {"latency_ms": latency_ms, "ttft_ms": ttft_ms}

    def _extract_usage(self, item: Any) -> dict[str, int] | None:
        for payload in self._candidate_mappings(item):
            for path in (
                ("usage",),
                ("usage_metadata",),
                ("token_usage",),
                ("response_metadata", "token_usage"),
                ("llm_output", "token_usage"),
            ):
                value = self._extract_path(payload, path)
                if isinstance(value, Mapping):
                    usage = self._normalize_usage(value)
                    if usage is not None:
                        return usage
        return None

    def _normalize_usage(self, value: Mapping[str, Any]) -> dict[str, int] | None:
        input_tokens = self._first_int(value, ("input_tokens", "prompt_tokens", "tokens_in"))
        output_tokens = self._first_int(value, ("output_tokens", "completion_tokens", "tokens_out"))
        total_tokens = self._first_int(value, ("total_tokens",))

        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None

        if total_tokens is None:
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        reasoning_tokens = 0

        details = value.get("output_token_details")

        if isinstance(details, Mapping):
            candidate = details.get("reasoning")

            if isinstance(candidate, int):
                reasoning_tokens = candidate

        return {
            "input_tokens": input_tokens or 0,
            "output_tokens": output_tokens or 0,
            "total_tokens": total_tokens,
            "output_token_details": {
                "reasoning": reasoning_tokens,
            },
        }

    def _first_int(
        self,
        value: Mapping[str, Any],
        keys: tuple[str, ...],
    ) -> int | None:
        for key in keys:
            candidate = value.get(key)
            if isinstance(candidate, int):
                return candidate
        return None

    def _extract_text_field(
        self,
        item: Any,
        paths: tuple[tuple[str, ...], ...],
    ) -> str | None:
        for payload in self._candidate_mappings(item):
            for path in paths:
                value = self._extract_path(payload, path)
                if isinstance(value, str) and value:
                    return value
        return None

    def _candidate_mappings(self, item: Any) -> list[Mapping[str, Any]]:
        candidates: list[Mapping[str, Any]] = []

        def collect(value: Any) -> None:
            if value is None:
                return

            mapping = self._as_mapping(value)

            if mapping is not None:
                candidates.append(mapping)

                for child in mapping.values():
                    collect(child)

            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(item)

        return candidates

    def _subvalue(self, item: Any, key: str) -> Any:
        if isinstance(item, Mapping):
            return item.get(key)
        return None

    def _as_mapping(self, value: Any) -> Mapping[str, Any] | None:
        if isinstance(value, Mapping):
            return value
        if is_dataclass(value):
            return asdict(value)
        if hasattr(value, "model_dump"):
            dumped = value.model_dump()
            if isinstance(dumped, Mapping):
                return dumped
        if hasattr(value, "dict"):
            dumped = value.dict()
            if isinstance(dumped, Mapping):
                return dumped
        return None

    def _extract_text_from_value(self, value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            for key in ("content", "output", "response", "text", "token", "delta", "message"):
                candidate = value.get(key)
                if isinstance(candidate, str) and candidate:
                    return candidate
            if "messages" in value:
                return self._text_from_messages(value["messages"])
            return None
        if isinstance(value, list):
            parts = [text for item in value if (text := self._extract_text_from_value(item))]
            return parts[-1] if parts else None
        if isinstance(value, tuple):
            parts = [text for item in value if (text := self._extract_text_from_value(item))]
            return parts[-1] if parts else None
        if hasattr(value, "content"):
            content = getattr(value, "content")
            if isinstance(content, str):
                return content
        if hasattr(value, "model_dump"):
            return self._extract_text_from_value(value.model_dump())
        if hasattr(value, "dict"):
            return self._extract_text_from_value(value.dict())
        return None

    def _extract_path(self, payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
        current: Any = payload
        for key in path:
            if not isinstance(current, Mapping) or key not in current:
                return None
            current = current[key]
        return current

    def _event(self, name: str, payload: dict[str, Any]) -> GenAILitEvent:
        return GenAILitEvent(name=name, payload=payload)

    def _serializable(self, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): self._serializable(val) for key, val in value.items()}
        if isinstance(value, list):
            return [self._serializable(item) for item in value]
        if isinstance(value, tuple):
            return [self._serializable(item) for item in value]
        if is_dataclass(value):
            return self._serializable(asdict(value))
        if hasattr(value, "model_dump"):
            return self._serializable(value.model_dump())
        return repr(value)
