from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genailit import GenAILitApp
from genailit.adapters.langgraph import LangGraphAdapter


class DemoState(TypedDict):
    messages: list[str]


def build_graph() -> Any:
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise SystemExit(
            'This example requires LangGraph. Install it with: pip install "genailit[langgraph]"'
        ) from exc

    def assistant(state: DemoState) -> DemoState:
        messages = list(state.get("messages", []))
        messages.append("Hola desde LangGraph + GenAILit")
        return {"messages": messages}

    workflow = StateGraph(DemoState)
    workflow.add_node("assistant", assistant)
    workflow.add_edge(START, "assistant")
    workflow.add_edge("assistant", END)

    return workflow.compile()


def build_app() -> GenAILitApp:
    graph = build_graph()
    adapter = LangGraphAdapter(graph)
    return GenAILitApp(adapter=adapter)


def main() -> None:
    app = build_app()
    app.run(host="0.0.0.0", port=8501)


if __name__ == "__main__":
    main()
