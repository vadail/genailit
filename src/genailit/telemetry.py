from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Any

from .events import GenAILitEvent
from .pricing import estimate_cost_usd


@dataclass(frozen=True, slots=True)
class _TelemetryRecord:
    event: GenAILitEvent
    timestamp: float
    session_id: str | None = None
    run_id: str | None = None


class TelemetryStore:
    def __init__(self) -> None:
        self._records: list[_TelemetryRecord] = []

    def record(
        self,
        event: GenAILitEvent,
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self._records.append(
            _TelemetryRecord(
                event=event,
                timestamp=time.perf_counter(),
                session_id=self._resolve_session_id(event, session_id),
                run_id=run_id,
            )
        )

    def extend(
        self,
        events: Iterable[GenAILitEvent],
        *,
        session_id: str | None = None,
        run_id: str | None = None,
    ) -> None:
        for event in events:
            self.record(event, session_id=session_id, run_id=run_id)

    def snapshot(self) -> tuple[GenAILitEvent, ...]:
        return tuple(record.event for record in self._records)

    def clear(self) -> None:
        self._records.clear()

    def get_session_trace(self, session_id: str) -> tuple[GenAILitEvent, ...]:
        return tuple(record.event for record in self._session_records(session_id))

    def get_session_metrics(self, session_id: str) -> dict[str, Any]:
        records = self._session_records(session_id)
        if not records:
            return {
                "total_events": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "model": None,
                "provider": None,
                "latency_ms": None,
                "ttft_ms": None,
                "cost_usd": None,
                "error_count": 0,
                "retry_count": 0,
                "estimated": {"tokens": False, "cost": False},
            }

        input_tokens, input_explicit = self._resolve_input_tokens(records)

        output_tokens, output_explicit = self._resolve_output_tokens(records)

        reasoning_tokens = self._resolve_reasoning_tokens(records)

        visible_output_tokens = max(
            output_tokens - reasoning_tokens,
            0,
        )

        total_tokens, _ = self._resolve_total_tokens(
            records,
            input_tokens,
            output_tokens,
        )

        provider = self._resolve_text_field(
            records,
            (
                ("provider",),
                ("metadata", "ls_provider"),
                ("metadata", "provider"),
            ),
        )
        model = self._resolve_text_field(
            records,
            (
                ("model",),
                ("metadata", "ls_model_name"),
                ("response_metadata", "model_name"),
                ("response_metadata", "model"),
            ),
        )
        latency_ms = self._resolve_float_field(
            records,
            (
                ("timing", "latency_ms"),
                ("latency_ms",),
            ),
        )
        ttft_ms = self._resolve_float_field(
            records,
            (
                ("timing", "ttft_ms"),
                ("ttft_ms",),
            ),
        )
        if latency_ms is None:
            latency_ms = self._estimate_latency_ms(records)
        if ttft_ms is None:
            ttft_ms = self._estimate_ttft_ms(records)
        cost_usd = self._resolve_float_field(
            records,
            (
                ("cost", "cost_usd"),
                ("cost_usd",),
            ),
        )

        tokens_estimated = not (input_explicit and output_explicit)
        if cost_usd is None:
            cost_usd = estimate_cost_usd(
                provider,
                model,
                input_tokens,
                output_tokens,
            )

        cost_estimated = cost_usd is not None

        return {
            "total_events": len(records),
            "input_tokens": input_tokens,
            "output_tokens": visible_output_tokens,
            "reasoning_tokens": reasoning_tokens,
            "total_tokens": total_tokens,
            "model": model,
            "provider": provider,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "cost_usd": cost_usd,
            "error_count": sum(
                1
                for record in records
                if record.event.name == "error"
            ),
            "retry_count": sum(
                self._retry_contribution(record)
                for record in records
            ),
            "estimated": {
                "tokens": tokens_estimated,
                "cost": cost_estimated,
            },
        }

    def _session_records(self, session_id: str) -> list[_TelemetryRecord]:
        return [
            record
            for record in self._records
            if self._resolve_session_id(record.event, record.session_id) == session_id
        ]

    def _resolve_session_id(
        self,
        event: GenAILitEvent,
        session_id: str | None,
    ) -> str | None:
        if session_id is not None:
            return session_id
        payload_session_id = event.payload.get("session_id")
        if isinstance(payload_session_id, str):
            return payload_session_id
        return None

    def _resolve_input_tokens(self, records: list[_TelemetryRecord]) -> tuple[int, bool]:
        value: int | None = None
        for record in records:
            candidate = self._extract_int_from_paths(
                record.event.payload,
                (
                    ("usage", "input_tokens"),
                    ("input_tokens",),
                    ("prompt_tokens",),
                    ("tokens_in",),
                ),
            )
            if candidate is not None:
                value = candidate
        if value is not None:
            return value, True
        return 0, False

    def _resolve_output_tokens(self, records: list[_TelemetryRecord]) -> tuple[int, bool]:
        value: int | None = None
        for record in records:
            candidate = self._extract_int_from_paths(
                record.event.payload,
                (
                    ("usage", "output_tokens"),
                    ("output_tokens",),
                    ("completion_tokens",),
                    ("tokens_out",),
                ),
            )
            if candidate is not None:
                value = candidate
        if value is not None:
            return value, True

        estimated = self._estimate_output_tokens(records)
        return estimated, False

    def _resolve_reasoning_tokens(
        self,
        records: list[_TelemetryRecord],
    ) -> int:
        value = 0

        for record in records:
            candidate = self._extract_int_from_paths(
                record.event.payload,
                (
                    ("usage", "output_token_details", "reasoning"),
                    ("usage", "reasoning_tokens"),
                ),
            )

            if candidate is not None:
                value = candidate

        return value

    def _resolve_total_tokens(
        self,
        records: list[_TelemetryRecord],
        input_tokens: int,
        output_tokens: int,
    ) -> tuple[int, bool]:
        total_tokens: int | None = None
        explicit = False
        for record in records:
            candidate = self._extract_int_from_paths(
                record.event.payload,
                (("usage", "total_tokens"), ("total_tokens",)),
            )
            if candidate is not None:
                total_tokens = candidate
                explicit = True
        if total_tokens is not None:
            return total_tokens, explicit
        return input_tokens + output_tokens, False

    def _resolve_text_field(
        self,
        records: list[_TelemetryRecord],
        paths: tuple[tuple[str, ...], ...],
    ) -> str | None:
        value: str | None = None
        for record in records:
            candidate = self._extract_str_from_paths(record.event.payload, paths)
            if candidate is not None:
                value = candidate
        return value

    def _resolve_float_field(
        self,
        records: list[_TelemetryRecord],
        paths: tuple[tuple[str, ...], ...],
    ) -> float | None:
        for record in records:
            candidate = self._extract_float_from_paths(record.event.payload, paths)
            if candidate is not None:
                return candidate
        return None

    def _estimate_output_tokens(self, records: list[_TelemetryRecord]) -> int:
        token_events = [record for record in records if record.event.name == "agent.token"]
        if token_events:
            return sum(
                self._count_text_tokens(self._extract_text(record.event.payload))
                for record in token_events
            )

        message_events = [record for record in records if record.event.name == "agent.message"]
        return sum(
            self._count_text_tokens(self._extract_text(record.event.payload))
            for record in message_events
        )

    def _estimate_latency_ms(self, records: list[_TelemetryRecord]) -> float:
        return (records[-1].timestamp - records[0].timestamp) * 1000.0

    def _estimate_ttft_ms(self, records: list[_TelemetryRecord]) -> float | None:
        first_timestamp = records[0].timestamp
        for record in records:
            if record.event.name in {"agent.token", "agent.message"}:
                return (record.timestamp - first_timestamp) * 1000.0
        return None

    def _extract_int_from_paths(
        self,
        payload: dict[str, Any],
        paths: tuple[tuple[str, ...], ...],
    ) -> int | None:
        for path in paths:
            value = self._extract_path(payload, path)
            if isinstance(value, int):
                return value
        return None

    def _extract_float_from_paths(
        self,
        payload: dict[str, Any],
        paths: tuple[tuple[str, ...], ...],
    ) -> float | None:
        for path in paths:
            value = self._extract_path(payload, path)
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _extract_str_from_paths(
        self,
        payload: dict[str, Any],
        paths: tuple[tuple[str, ...], ...],
    ) -> str | None:
        for path in paths:
            value = self._extract_path(payload, path)
            if isinstance(value, str) and value:
                return value
        return None

    def _extract_path(self, payload: Any, path: tuple[str, ...]) -> Any:
        current: Any = payload
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _extract_text(self, payload: dict[str, Any]) -> str:
        for key in ("delta", "token", "content", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
        return ""

    def _count_text_tokens(self, text: str) -> int:
        stripped = text.strip()
        if not stripped:
            return 0
        return len(stripped.split())

    def _retry_contribution(self, record: _TelemetryRecord) -> int:
        event_retry = 1 if "retry" in record.event.name.lower() else 0
        payload_retry = record.event.payload.get("retry_count")
        if isinstance(payload_retry, int):
            return max(event_retry, payload_retry)
        return event_retry
