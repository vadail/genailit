from __future__ import annotations

import inspect
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Callable
from uuid import uuid4

from ..events import GenAILitEvent
from .base import AdapterContext, BaseAgentAdapter


AgentFunction = Callable[[Any, AdapterContext], Any]


@dataclass(slots=True)
class FunctionAgentAdapter(BaseAgentAdapter):
    func: AgentFunction

    async def stream(
        self,
        input_data: Any,
        context: AdapterContext | None = None,
    ) -> AsyncIterator[GenAILitEvent]:
        runtime_context = context or self._build_context(input_data)
        for event in self._session_start(runtime_context):
            yield event

        saw_session_ended = False
        try:
            result = self.func(input_data, runtime_context)
            if inspect.isawaitable(result):
                result = await self._await_result(result)

            async for event in self._iterate_result(result):
                if event.name == "session.started":
                    continue
                if event.name == "session.ended":
                    saw_session_ended = True
                yield event
        except Exception as exc:
            yield GenAILitEvent(
                name="error",
                payload={"message": str(exc), "type": exc.__class__.__name__},
            )
        finally:
            if not saw_session_ended:
                for event in self._session_end(runtime_context):
                    yield event

    def _build_context(self, input_data: Any) -> AdapterContext:
        return AdapterContext(session_id=uuid4().hex, run_id=uuid4().hex, input_data=input_data)

    async def _await_result(self, result: Awaitable[Any]) -> Any:
        return await result

    async def _iterate_result(self, result: Any) -> AsyncIterator[GenAILitEvent]:
        if result is None:
            return
        if isinstance(result, GenAILitEvent):
            yield result
            return
        if isinstance(result, str):
            yield GenAILitEvent(name="agent.token", payload={"delta": result})
            return
        if isinstance(result, Mapping):
            yield self._coerce_mapping(result)
            return
        if isinstance(result, AsyncIterable):
            async for item in result:
                yield self._coerce_item(item)
            return
        if isinstance(result, Iterable):
            for item in result:
                yield self._coerce_item(item)
            return
        yield GenAILitEvent(name="agent.message", payload={"content": str(result)})

    def _coerce_item(self, item: Any) -> GenAILitEvent:
        if isinstance(item, GenAILitEvent):
            return item
        if isinstance(item, str):
            return GenAILitEvent(name="agent.token", payload={"delta": item})
        if isinstance(item, Mapping):
            return self._coerce_mapping(item)
        return GenAILitEvent(name="agent.message", payload={"content": str(item)})

    def _coerce_mapping(self, item: Mapping[str, Any]) -> GenAILitEvent:
        name = item.get("name")
        payload = item.get("payload")
        if isinstance(name, str) and isinstance(payload, Mapping):
            return GenAILitEvent(name=name, payload=dict(payload))

        if isinstance(item.get("delta"), str):
            return GenAILitEvent(name="agent.token", payload={"delta": str(item["delta"])})
        if isinstance(item.get("token"), str):
            return GenAILitEvent(name="agent.token", payload={"delta": str(item["token"])})
        if isinstance(item.get("content"), str):
            return GenAILitEvent(name="agent.message", payload={"content": str(item["content"])})
        if isinstance(item.get("message"), str):
            return GenAILitEvent(name="agent.message", payload={"content": str(item["message"])})

        return GenAILitEvent(name="agent.message", payload={"content": str(dict(item))})

    def _session_start(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
        return (
            GenAILitEvent(
                name="session.started",
                payload={"session_id": context.session_id, "run_id": context.run_id},
            ),
        )

    def _session_end(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
        return (
            GenAILitEvent(
                name="session.ended",
                payload={"session_id": context.session_id, "run_id": context.run_id},
            ),
        )
