from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


@dataclass(slots=True)
class SystemManifest:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)
    mermaid: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "tools": self.tools,
            "mcp_servers": self.mcp_servers,
            "mermaid": self.mermaid,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }


def introspect_system(adapter_or_graph: Any) -> dict[str, Any]:
    manifest = SystemManifest(metadata={"source": "introspection"})
    graph = getattr(adapter_or_graph, "graph", adapter_or_graph)

    try:
        graph_view = graph.get_graph() if hasattr(graph, "get_graph") else graph
    except Exception as exc:
        manifest.warnings.append(f"get_graph failed: {exc}")
        graph_view = graph

    try:
        _extract_mermaid(graph_view, manifest)
        _extract_nodes(graph_view, manifest)
        _extract_edges(graph_view, manifest)
    except Exception as exc:
        manifest.warnings.append(f"graph introspection failed: {exc}")

    if not manifest.nodes and not manifest.edges and manifest.mermaid is None:
        manifest.warnings.append("No system map available")

    return manifest.to_dict()


def _extract_mermaid(graph_view: Any, manifest: SystemManifest) -> None:
    draw_mermaid = getattr(graph_view, "draw_mermaid", None)
    if not callable(draw_mermaid):
        return
    try:
        mermaid = draw_mermaid()
    except Exception as exc:
        manifest.warnings.append(f"draw_mermaid failed: {exc}")
        return
    if isinstance(mermaid, str) and mermaid:
        manifest.mermaid = mermaid


def _extract_nodes(graph_view: Any, manifest: SystemManifest) -> None:
    raw_nodes = getattr(graph_view, "nodes", None)
    if raw_nodes is None and isinstance(graph_view, Mapping):
        raw_nodes = graph_view.get("nodes")
    for node_id, node_value in _iter_named_items(raw_nodes):
        node = _node_entry(node_id, node_value)
        manifest.nodes.append(node)
        tools = _detect_tools(node_value, node_id=node["id"])
        servers = _detect_mcp_servers(node_value, node_id=node["id"])
        node["tools"] = [tool["name"] for tool in tools if tool.get("name")]
        manifest.tools.extend(tools)
        manifest.mcp_servers.extend(servers)


def _extract_edges(graph_view: Any, manifest: SystemManifest) -> None:
    raw_edges = getattr(graph_view, "edges", None)
    if raw_edges is None and isinstance(graph_view, Mapping):
        raw_edges = graph_view.get("edges")
    for edge in _iter_edges(raw_edges):
        manifest.edges.append(edge)


def _iter_named_items(value: Any) -> list[tuple[str, Any]]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [(str(key), item) for key, item in value.items()]
    items: list[tuple[str, Any]] = []
    if isinstance(value, (list, tuple, set)):
        for index, item in enumerate(value):
            name = _name_of(item) or str(index)
            items.append((name, item))
        return items
    return []


def _iter_edges(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    edges: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        iterable = value.values()
    else:
        iterable = value if isinstance(value, (list, tuple, set)) else ()
    for item in iterable:
        edge = _edge_entry(item)
        if edge is not None:
            edges.append(edge)
    return edges


def _node_entry(node_id: str, value: Any) -> dict[str, Any]:
    return {
        "id": node_id,
        "name": _name_of(value) or node_id,
        "metadata": _metadata_of(value),
    }


def _edge_entry(value: Any) -> dict[str, Any] | None:
    if isinstance(value, Mapping):
        source = value.get("source") or value.get("from") or value.get("start")
        target = value.get("target") or value.get("to") or value.get("end")
        if source is not None and target is not None:
            return {"source": str(source), "target": str(target)}
    if isinstance(value, tuple | list) and len(value) >= 2:
        return {"source": str(value[0]), "target": str(value[1])}
    source = getattr(value, "source", None) or getattr(value, "start", None)
    target = getattr(value, "target", None) or getattr(value, "end", None)
    if source is not None and target is not None:
        return {"source": str(source), "target": str(target)}
    return None


def _detect_tools(value: Any, *, node_id: str) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for candidate in _tool_candidates(value):
        if candidate is None:
            continue
        for item in candidate if isinstance(candidate, (list, tuple, set)) else (candidate,):
            name = _name_of(item)
            if name is not None:
                tools.append({"name": name, "node_id": node_id})
    return _dedupe(tools, keys=("node_id", "name"))


def _detect_mcp_servers(value: Any, *, node_id: str) -> list[dict[str, Any]]:
    servers: list[dict[str, Any]] = []
    for candidate in _mcp_candidates(value):
        mapping = _as_mapping(candidate)
        if mapping is None:
            continue
        server_url = _first_present(mapping, ("server_url", "url", "endpoint"))
        transport = _first_present(mapping, ("transport", "transport_type"))
        name = _first_present(mapping, ("name", "id")) or server_url or "mcp_server"
        servers.append(
            {
                "name": str(name),
                "node_id": node_id,
                "server_url": str(server_url) if server_url is not None else None,
                "transport": str(transport) if transport is not None else None,
            }
        )
    return _dedupe(servers, keys=("node_id", "name", "server_url"))


def _tool_candidates(value: Any) -> list[Any]:
    candidates = [_attr_or_mapping(value, key) for key in ("tools", "bound_tools", "tool")]
    kwargs = _attr_or_mapping(value, "kwargs")
    if isinstance(kwargs, Mapping):
        candidates.append(kwargs.get("tools"))
    return candidates


def _mcp_candidates(value: Any) -> list[Any]:
    candidates = [_attr_or_mapping(value, key) for key in ("mcp_servers", "mcp_server")]
    for key in ("client", "session"):
        candidate = _attr_or_mapping(value, key)
        if _looks_like_mcp(candidate):
            candidates.append(candidate)
    if _looks_like_mcp(value):
        candidates.append(value)
    expanded: list[Any] = []
    for candidate in candidates:
        if isinstance(candidate, (list, tuple, set)):
            expanded.extend(candidate)
        else:
            expanded.append(candidate)
    return expanded


def _looks_like_mcp(value: Any) -> bool:
    mapping = _as_mapping(value)
    if mapping is None:
        return False
    return any(key in mapping for key in ("server_url", "transport", "client", "session"))


def _name_of(value: Any) -> str | None:
    mapping = _as_mapping(value)
    if mapping is not None:
        for key in ("name", "id"):
            item = mapping.get(key)
            if isinstance(item, str) and item:
                return item
        lc_id = mapping.get("lc_id")
        if isinstance(lc_id, (list, tuple)) and lc_id:
            return str(lc_id[-1])
        if isinstance(lc_id, str) and lc_id:
            return lc_id
    for key in ("name", "id"):
        item = getattr(value, key, None)
        if isinstance(item, str) and item:
            return item
    return None


def _metadata_of(value: Any) -> dict[str, Any]:
    mapping = _as_mapping(value)
    metadata = mapping.get("metadata") if mapping is not None else getattr(value, "metadata", None)
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def _attr_or_mapping(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    if hasattr(value, "dict"):
        dumped = value.dict()
        return dumped if isinstance(dumped, Mapping) else None
    if hasattr(value, "__dict__"):
        return vars(value)
    return None


def _first_present(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _dedupe(items: list[dict[str, Any]], *, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        marker = tuple(item.get(key) for key in keys)
        if marker in seen:
            continue
        seen.add(marker)
        result.append(item)
    return result
