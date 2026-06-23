from __future__ import annotations

from genailit import GenAILitApp, introspect_system
from genailit.adapters.langgraph import LangGraphAdapter


class FakeTool:
    def __init__(self, name: str) -> None:
        self.name = name


class FakeNode:
    def __init__(self) -> None:
        self.name = "planner"
        self.tools = [FakeTool("search")]
        self.bound_tools = [FakeTool("calculator")]
        self.mcp_servers = [{"name": "filesystem", "server_url": "http://mcp.local", "transport": "sse"}]


class FakeGraphView:
    nodes = {"planner": FakeNode(), "writer": {"name": "writer"}}
    edges = [("planner", "writer")]

    def draw_mermaid(self) -> str:
        return "graph TD\n  planner --> writer"


class FakeLangGraph:
    def get_graph(self) -> FakeGraphView:
        return FakeGraphView()


def test_introspection_extracts_langgraph_nodes_and_edges() -> None:
    manifest = introspect_system(FakeLangGraph())

    assert [node["id"] for node in manifest["nodes"]] == ["planner", "writer"]
    assert manifest["edges"] == [{"source": "planner", "target": "writer"}]


def test_introspection_captures_mermaid_fallback() -> None:
    manifest = introspect_system(FakeLangGraph())

    assert manifest["mermaid"] == "graph TD\n  planner --> writer"


def test_introspection_detects_tools_from_common_attributes() -> None:
    manifest = introspect_system(FakeLangGraph())

    assert {"name": "search", "node_id": "planner"} in manifest["tools"]
    assert {"name": "calculator", "node_id": "planner"} in manifest["tools"]


def test_introspection_detects_mcp_servers_from_common_attributes() -> None:
    manifest = introspect_system(FakeLangGraph())

    assert manifest["mcp_servers"] == [
        {
            "name": "filesystem",
            "node_id": "planner",
            "server_url": "http://mcp.local",
            "transport": "sse",
        }
    ]


def test_introspection_never_fails_when_graph_is_unknown() -> None:
    manifest = introspect_system(object())

    assert manifest["nodes"] == []
    assert manifest["edges"] == []
    assert "No system map available" in manifest["warnings"]


def test_langgraph_adapter_exposes_auto_system_manifest() -> None:
    adapter = LangGraphAdapter(FakeLangGraph())

    manifest = adapter.get_system_manifest()

    assert manifest["nodes"][0]["id"] == "planner"


def test_genailit_app_uses_adapter_system_manifest() -> None:
    class ManifestAdapter:
        def get_system_manifest(self) -> dict[str, object]:
            return {
                "nodes": [{"id": "agent-a", "name": "Agent A"}],
                "edges": [],
                "tools": [],
                "mcp_servers": [],
                "mermaid": None,
                "metadata": {},
                "warnings": [],
            }

    app = GenAILitApp(adapter=ManifestAdapter())  # type: ignore[arg-type]

    assert '"Agent A"' in app._render_html()
