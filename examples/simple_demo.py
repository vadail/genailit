from __future__ import annotations

import sys
import time
from pathlib import Path
from collections.abc import Iterable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genailit import AdapterContext, BaseAgentAdapter, GenAILitApp, GenAILitEvent


class DemoAdapter(BaseAgentAdapter):
    def build_events(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
        del context
        return ()

    def stream(self, input_data: object, context: AdapterContext) -> Iterable[GenAILitEvent]:
        del input_data, context
        yield from self._emit("session.started")
        yield from self._emit("agent.thinking")
        yield from self._emit("agent.token", {"token": "Hola "})
        yield from self._emit("agent.token", {"token": "desde "})
        yield from self._emit("agent.token", {"token": "GenAILit"})
        yield from self._emit("tool.started")
        yield from self._emit("tool.ended")
        yield from self._emit("metrics.updated")
        yield from self._emit("session.ended")

    def _emit(self, name: str, payload: dict[str, object] | None = None) -> Iterable[GenAILitEvent]:
        time.sleep(0.15)
        yield GenAILitEvent(name=name, payload={} if payload is None else payload)


if __name__ == "__main__":
    app = GenAILitApp(adapter=DemoAdapter())
    app.run()

