from __future__ import annotations

import builtins
import importlib.util
import sys
import types
from pathlib import Path

from genailit.adapters.langgraph import LangGraphAdapter


def _load_demo_module():
    module_path = Path(__file__).resolve().parents[1] / "examples" / "langgraph_demo.py"
    spec = importlib.util.spec_from_file_location("langgraph_demo", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_langgraph_demo_builds_graph_and_runs_app(monkeypatch) -> None:
    compiled_graph = object()

    class FakeStateGraph:
        def __init__(self, state_type: object) -> None:
            self.state_type = state_type
            self.nodes: list[tuple[str, object]] = []
            self.edges: list[tuple[object, object]] = []

        def add_node(self, name: str, fn: object) -> None:
            self.nodes.append((name, fn))

        def add_edge(self, source: object, target: object) -> None:
            self.edges.append((source, target))

        def compile(self) -> object:
            return compiled_graph

    fake_graph_module = types.ModuleType("langgraph.graph")
    fake_graph_module.START = "START"
    fake_graph_module.END = "END"
    fake_graph_module.StateGraph = FakeStateGraph

    fake_langgraph_package = types.ModuleType("langgraph")
    fake_langgraph_package.graph = fake_graph_module

    monkeypatch.setitem(sys.modules, "langgraph", fake_langgraph_package)
    monkeypatch.setitem(sys.modules, "langgraph.graph", fake_graph_module)

    module = _load_demo_module()
    graph = module.build_graph()

    assert graph is compiled_graph

    recorded: dict[str, object] = {}

    class FakeApp:
        def __init__(self, adapter: object) -> None:
            recorded["adapter"] = adapter

        def run(self, host: str, port: int) -> None:
            recorded["host"] = host
            recorded["port"] = port

    monkeypatch.setattr(module, "GenAILitApp", FakeApp)

    module.main()

    adapter = recorded["adapter"]
    assert isinstance(adapter, LangGraphAdapter)
    assert adapter.graph is compiled_graph
    assert recorded["host"] == "0.0.0.0"
    assert recorded["port"] == 8501


def test_langgraph_demo_shows_clear_missing_dependency_error(monkeypatch) -> None:
    module = _load_demo_module()

    original_import = builtins.__import__

    def fake_import(name: str, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[override]
        if name == "langgraph.graph":
            raise ImportError("No module named langgraph.graph")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        module.build_graph()
    except SystemExit as exc:
        assert 'pip install "genailit[langgraph]"' in str(exc)
    else:
        raise AssertionError("Expected SystemExit when LangGraph is missing")

