from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from ..events import GenAILitEvent
from ..telemetry import TelemetryStore


@dataclass(frozen=True, slots=True)
class AdapterContext:
    session_id: str
    run_id: str | None = None
    input_data: Any = None
    include_raw: bool = False
    telemetry: TelemetryStore | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def with_metadata(self, **items: Any) -> "AdapterContext":
        merged = dict(self.metadata)
        merged.update(items)
        return AdapterContext(
            session_id=self.session_id,
            run_id=self.run_id,
            input_data=self.input_data,
            include_raw=self.include_raw,
            telemetry=self.telemetry,
            metadata=merged,
        )


class BaseAgentAdapter:
    def __new__(cls, *args: Any, **kwargs: Any) -> "BaseAgentAdapter":
        if cls is BaseAgentAdapter:
            raise TypeError("BaseAgentAdapter cannot be instantiated directly")
        return super().__new__(cls)

    def build_events(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
        raise NotImplementedError

    def stream(self, input_data: Any, context: AdapterContext) -> Iterable[GenAILitEvent]:
        del input_data
        return self.build_events(context)

    def run(self, context: AdapterContext) -> tuple[GenAILitEvent, ...]:
        events = tuple(self.stream(context.input_data, context))
        if context.telemetry is not None:
            context.telemetry.extend(events)
        return events
