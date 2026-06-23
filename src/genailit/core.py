from __future__ import annotations

import json
from collections.abc import AsyncIterable, Iterable
from typing import Any, Callable, overload
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from .adapters.base import AdapterContext, BaseAgentAdapter
from .adapters.function import FunctionAgentAdapter
from .debug import DebugPanel
from .events import GenAILitEvent
from .introspection import introspect_system
from .telemetry import TelemetryStore


_NO_AGENT_ERROR = (
    "GenAILitApp has no agent registered. "
    "Use @app.agent or pass adapter=... to GenAILitApp."
)


def _websocket_path_for_pathname(pathname: str) -> str:
    if pathname == "/":
        return "/ws"
    return pathname.rstrip("/") + "/ws"


class GenAILitApp:
    def __init__(self, adapter: BaseAgentAdapter | None = None, manifest: dict[str, Any] | None = None) -> None:
        self.adapter = adapter
        self.manifest = manifest
        self.telemetry = TelemetryStore()
        self._app = FastAPI()
        self._register_routes()

    @property
    def asgi_app(self) -> FastAPI:
        return self._app

    def run(self, host: str = "0.0.0.0", port: int = 8501, reload: bool = False) -> None:
        uvicorn.run(self._app, host=host, port=port, reload=reload)

    @overload
    def agent(self, func: None = None) -> Callable[[Callable[[Any, AdapterContext], Any]], Callable[[Any, AdapterContext], Any]]:
        ...

    @overload
    def agent(self, func: Callable[[Any, AdapterContext], Any]) -> Callable[[Any, AdapterContext], Any]:
        ...

    def agent(self, func: Callable[[Any, AdapterContext], Any] | None = None):
        if func is None:
            def decorator(target: Callable[[Any, AdapterContext], Any]) -> Callable[[Any, AdapterContext], Any]:
                self.adapter = FunctionAgentAdapter(func=target)
                return target

            return decorator

        self.adapter = FunctionAgentAdapter(func=func)
        return func

    def _register_routes(self) -> None:
        @self._app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        @self._app.get("/", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(self._render_html())

        @self._app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket) -> None:
            await websocket.accept()

            try:
                input_data = await websocket.receive_json()

                conversation_id = input_data.get("conversation_id")

                session_id = (
                    conversation_id
                    if conversation_id
                    else uuid4().hex
                )

                run_id = uuid4().hex

                if self.adapter is None:
                    raise RuntimeError(_NO_AGENT_ERROR)

                context = AdapterContext(
                    session_id=session_id,
                    run_id=run_id,
                    input_data=input_data,
                    telemetry=self.telemetry,
                )

                async for event in self._iterate_events(
                    self.adapter.stream(input_data, context)
                ):
                    await self._send_event(
                        websocket,
                        event,
                        session_id=session_id,
                        run_id=run_id,
                    )

            except WebSocketDisconnect:
                return

            except Exception as exc:
                await self._send_error(
                    websocket,
                    exc,
                    session_id=session_id,
                    run_id=run_id,
                )
                await websocket.close(code=1011)
               
    async def _iterate_events(
        self,
        stream: Iterable[GenAILitEvent] | AsyncIterable[GenAILitEvent],
    ) -> AsyncIterable[GenAILitEvent]:
        if isinstance(stream, AsyncIterable):
            async for event in stream:
                yield event
            return

        for event in stream:
            yield event

    async def _send_event(
        self,
        websocket: WebSocket,
        event: GenAILitEvent,
        *,
        session_id: str,
        run_id: str,
    ) -> None:
        self.telemetry.record(event, session_id=session_id, run_id=run_id)
        await websocket.send_json(event.model_dump(mode="json"))

    async def _send_error(
        self,
        websocket: WebSocket,
        exc: Exception,
        *,
        session_id: str,
        run_id: str,
    ) -> None:
        event = GenAILitEvent(
            name="error",
            payload={"message": str(exc), "type": exc.__class__.__name__},
        )
        await self._send_event(websocket, event, session_id=session_id, run_id=run_id)

    def _resolve_system_manifest(self) -> dict[str, Any]:
        if self.manifest is not None:
            return self.manifest
        if self.adapter is None:
            return self._empty_system_manifest()

        get_manifest = getattr(self.adapter, "get_system_manifest", None)
        if callable(get_manifest):
            try:
                manifest = get_manifest()
                if isinstance(manifest, dict):
                    return manifest
            except Exception as exc:
                return self._empty_system_manifest(f"adapter manifest failed: {exc}")

        graph = getattr(self.adapter, "graph", None)
        if graph is not None:
            return introspect_system(graph)

        return self._empty_system_manifest()

    def _empty_system_manifest(self, warning: str = "No system map available") -> dict[str, Any]:
        return {
            "nodes": [],
            "edges": [],
            "tools": [],
            "mcp_servers": [],
            "mermaid": None,
            "metadata": {},
            "warnings": [warning],
        }

    def _render_html(self) -> str:
        debug_panel = DebugPanel().render()
        system_manifest_json = json.dumps(self._resolve_system_manifest())
        html = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>GenAILit</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --panel: #ffffff;
      --panel-soft: #f1f5f9;
      --line: #d8dee8;
      --text: #172033;
      --muted: #64748b;
      --accent: #2563eb;
      --accent-strong: #1d4ed8;
      --success: #15803d;
      --danger: #b91c1c;
      --tool: #0f766e;
      --agent: #4338ca;
      --shadow: 0 18px 42px rgba(17, 24, 39, 0.12);
    }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    .app-shell {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      height: 100vh;
      box-sizing: border-box;
    }
    .app-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      padding: 0.8rem 1.25rem;
      background: rgba(255, 255, 255, 0.92);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(14px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 0.75rem;
      min-width: 0;
    }
    .brand-mark {
      display: grid;
      place-items: center;
      width: 2rem;
      height: 2rem;
      border-radius: 8px;
      background: var(--text);
      color: white;
      font-weight: 700;
    }
    .brand h1 {
      margin: 0;
      font-size: 1rem;
      line-height: 1.1;
    }
    .brand p, .eyebrow {
      margin: 0;
      color: var(--muted);
      font-size: 0.78rem;
    }
    .sr-only {
      position: absolute;
      width: 1px;
      height: 1px;
      padding: 0;
      margin: -1px;
      overflow: hidden;
      clip: rect(0, 0, 0, 0);
      white-space: nowrap;
      border: 0;
    }
    .header-actions {
      display: flex;
      align-items: center;
      gap: 0.6rem;
    }
    .chat-timeline {
      overflow-y: auto;
      padding: 1.4rem 1rem 2rem;
    }
    .timeline-inner {
      width: min(920px, 100%);
      margin: 0 auto;
      display: flex;
      flex-direction: column;
      gap: 0.85rem;
    }
    .empty-state {
      margin: 10vh auto 0;
      max-width: 620px;
      text-align: center;
      color: var(--muted);
    }
    .empty-state h2 {
      margin: 0 0 0.5rem;
      color: var(--text);
      font-size: clamp(1.7rem, 4vw, 2.4rem);
    }
    .message-row {
      display: flex;
      align-items: flex-start;
      gap: 0.65rem;
    }
    .message-row.user {
      justify-content: flex-end;
    }
    .message-row.assistant {
      justify-content: flex-start;
    }
    .message {
      max-width: min(720px, 82%);
      padding: 0.85rem 1rem;
      border-radius: 12px;
      line-height: 1.5;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message.user {
      background: var(--accent);
      color: white;
      border-bottom-right-radius: 4px;
    }
    .message.assistant {
      background: var(--panel);
      border: 1px solid var(--line);
      border-bottom-left-radius: 4px;
      box-shadow: 0 8px 22px rgba(17, 24, 39, 0.06);
    }
    .message.final {
      border-color: rgba(37, 99, 235, 0.35);
      box-shadow: 0 10px 28px rgba(37, 99, 235, 0.10);
    }
    .event-card, details.event-card {
      max-width: min(760px, 92%);
      padding: 0.75rem 0.9rem;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel);
      box-shadow: 0 6px 18px rgba(17, 24, 39, 0.05);
    }
    .event-card.agent {
      border-left: 4px solid var(--agent);
    }
    .event-card.tool {
      border-left: 4px solid var(--tool);
    }
    .event-card.error {
      border-left: 4px solid var(--danger);
    }
    .card-title {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 0.75rem;
      font-weight: 650;
    }
    .card-body {
      margin-top: 0.45rem;
      color: var(--muted);
      font-size: 0.92rem;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 0.18rem 0.5rem;
      color: white;
      background: var(--agent);
      font-size: 0.76rem;
      line-height: 1.2;
      white-space: nowrap;
    }
    .badge.tool {
      background: var(--tool);
    }
    .badge.node {
      background: #475569;
    }
    .badge.success {
      background: var(--success);
    }
    .badge.error {
      background: var(--danger);
    }
    .handoff {
      align-self: center;
      padding: 0.45rem 0.75rem;
      border: 1px dashed #9aa8bd;
      border-radius: 999px;
      background: #ffffff;
      color: #334155;
      font-size: 0.88rem;
    }
    details.reasoning summary, details.event-card summary, #raw-events-details summary {
      cursor: pointer;
      font-weight: 650;
    }
    .composer {
      border-top: 1px solid var(--line);
      background: rgba(247, 248, 251, 0.94);
      padding: 0.85rem 1rem 1rem;
      position: sticky;
      bottom: 0;
    }
    .composer-inner {
      width: min(920px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 0.75rem;
      align-items: end;
    }
    textarea {
      width: 100%;
      box-sizing: border-box;
      min-height: 4.25rem;
      max-height: 12rem;
      resize: vertical;
      border-radius: 10px;
      border: 1px solid var(--line);
      padding: 0.9rem;
      font: inherit;
      background: white;
      color: var(--text);
    }
    button {
      border: 0;
      border-radius: 10px;
      padding: 0.72rem 1rem;
      background: var(--accent);
      color: white;
      cursor: pointer;
      font: inherit;
      font-weight: 650;
    }
    button:hover {
      background: var(--accent-strong);
    }
    .ghost-button {
      background: white;
      color: var(--text);
      border: 1px solid var(--line);
    }
    .ghost-button:hover {
      background: var(--panel-soft);
    }
    pre {
      background: #111827;
      color: #dbeafe;
      padding: 1rem;
      white-space: pre-wrap;
      overflow-x: auto;
      border-radius: 10px;
      min-height: 1.5rem;
    }
    .telemetry-backdrop {
      position: fixed;
      inset: 0;
      background: rgba(15, 23, 42, 0.28);
      z-index: 30;
    }
    .telemetry-drawer {
      position: fixed;
      top: 0;
      right: 0;
      z-index: 40;
      width: min(420px, 100vw);
      height: 100vh;
      box-sizing: border-box;
      padding: 1rem;
      overflow-y: auto;
      background: white;
      border-left: 1px solid var(--line);
      box-shadow: var(--shadow);
      transform: translateX(105%);
      transition: transform 180ms ease;
    }
    .telemetry-drawer.open {
      transform: translateX(0);
    }
    .drawer-header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 1rem;
      margin-bottom: 1rem;
    }
    .drawer-header h2 {
      margin: 0;
    }
    .metrics-grid {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.4rem 0.75rem;
    }
    .metrics-grid dd {
      margin: 0;
      font-weight: 600;
    }
    .system-list {
      display: grid;
      gap: 0.45rem;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .system-item {
      padding: 0.55rem 0.65rem;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-soft);
      color: var(--text);
      overflow-wrap: anywhere;
    }
    .legacy-output {
      display: none;
    }
    @media (max-width: 720px) {
      .composer-inner {
        grid-template-columns: 1fr;
      }
      .message {
        max-width: 92%;
      }
      .header-actions {
        gap: 0.4rem;
      }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <header class="app-header">
      <div class="brand">
        <div class="brand-mark">G</div>
        <div>
          <h1>GenAILit</h1>
          <p>Multi-agent conversation workspace</p>
        </div>
      </div>
      <div class="header-actions">
        <button id="system-toggle" class="ghost-button" type="button">System</button>
        <button id="telemetry-toggle" class="ghost-button" type="button">Telemetry</button>
      </div>
    </header>

    <main id="timeline" class="chat-timeline" aria-live="polite">
      <div id="timeline-inner" class="timeline-inner">
        <section id="empty-state" class="empty-state">
          <h2>Start a multi-agent run</h2>
          <p>Send a message and watch responses, reasoning summaries, handoffs, tools, and telemetry unfold as events arrive.</p>
        </section>
      </div>
    </main>

    <form id="composer" class="composer">
      <div class="composer-inner">
        <label class="sr-only" for="input">Prompt input</label>
        <textarea id="input" rows="3" placeholder="Escribe tu mensaje..."></textarea>
        <button id="run" type="submit">Send</button>
      </div>
    </form>

    <pre id="token" class="legacy-output" aria-hidden="true"></pre>
    <pre id="events" class="legacy-output" aria-hidden="true"></pre>
    __DEBUG_PANEL__
  </div>

  <script>
    const SYSTEM_MANIFEST = __SYSTEM_MANIFEST__;

    document.addEventListener("DOMContentLoaded", () => {
      const runButton = document.getElementById("run");
      const composer = document.getElementById("composer");
      const input = document.getElementById("input");
      const token = document.getElementById("token");
      const events = document.getElementById("events");
      const timeline = document.getElementById("timeline");
      const timelineInner = document.getElementById("timeline-inner");
      const emptyState = document.getElementById("empty-state");
      const telemetryToggle = document.getElementById("telemetry-toggle");
      const telemetryClose = document.getElementById("telemetry-close");
      const telemetryDrawer = document.getElementById("telemetry-drawer");
      const telemetryBackdrop = document.getElementById("telemetry-backdrop");
      const systemToggle = document.getElementById("system-toggle");
      const systemClose = document.getElementById("system-close");
      const systemDrawer = document.getElementById("system-drawer");
      const systemBackdrop = document.getElementById("system-backdrop");
      const systemNodeCount = document.getElementById("system-node-count");
      const systemToolCount = document.getElementById("system-tool-count");
      const systemMcpCount = document.getElementById("system-mcp-count");
      const systemNodes = document.getElementById("system-nodes");
      const systemEdges = document.getElementById("system-edges");
      const systemTools = document.getElementById("system-tools");
      const systemMcpServers = document.getElementById("system-mcp-servers");
      const systemMermaid = document.getElementById("system-mermaid");
      const systemWarnings = document.getElementById("system-warnings");
      const debugEvents = document.getElementById("debug-events");
      const debugTree = document.getElementById("debug-tree");

      const metricTotalEvents = document.getElementById("metric-total-events");
      const metricInputTokens = document.getElementById("metric-input-tokens");
      const metricReasoningTokens = document.getElementById("metric-reasoning-tokens");
      const metricOutputTokens = document.getElementById("metric-output-tokens");
      const metricTotalTokens = document.getElementById("metric-total-tokens");
      const metricProvider = document.getElementById("metric-provider");
      const metricModel = document.getElementById("metric-model");
      const metricCostUsd = document.getElementById("metric-cost-usd");
      const metricLatency = document.getElementById("metric-latency");
      const metricTtft = document.getElementById("metric-ttft");
      const metricErrorCount = document.getElementById("metric-error-count");
      const metricRetryCount = document.getElementById("metric-retry-count");
      const metricEstimatedTokens = document.getElementById("metric-estimated-tokens");
      const metricEstimatedCost = document.getElementById("metric-estimated-cost");

      let socket = null;
      let totalEvents = 0;
      let inputTokens = 0;
      let outputTokens = 0;
      let reasoningTokens = 0;
      let totalTokens = 0;
      let provider = "-";
      let model = "-";
      let costUsd = null;
      let latencyMs = null;
      let ttftMs = null;
      let errorCount = 0;
      let retryCount = 0;
      let estimatedTokens = true;
      let estimatedCost = true;
      let sawRealUsage = false;
      let sawAgentToken = false;
      let runStartedAt = null;
      let firstTokenAt = null;
      let tree = [];
      let stack = [];
      let currentAssistantMessage = null;
      let activeTools = new Map();
      let sawTokenInCurrentRun = false;

      let conversationId =
        localStorage.getItem("genailit_conversation_id");

      if (!conversationId) {
        conversationId = crypto.randomUUID();
        localStorage.setItem(
          "genailit_conversation_id",
          conversationId
        );
      }

      function appendEvent(value) {
        events.textContent += value + "\\n";
        if (debugEvents) {
          debugEvents.textContent += value + "\\n";
        }
      }

      function countTokens(text) {
        if (typeof text !== "string") return 0;
        const trimmed = text.trim();
        if (!trimmed) return 0;
        return trimmed.split(/\\s+/).length;
      }

      function scrollTimeline() {
        timeline.scrollTop = timeline.scrollHeight;
      }

      function hideEmptyState() {
        if (emptyState) {
          emptyState.hidden = true;
        }
      }

      function truncateText(value, maxLength) {
        if (typeof value !== "string") return "";
        if (value.length <= maxLength) return value;
        return value.slice(0, maxLength - 1) + "...";
      }

      function renderList(target, items, emptyText, formatter) {
        target.textContent = "";
        if (!items || items.length === 0) {
          target.textContent = emptyText;
          return;
        }
        items.forEach((item) => {
          const row = document.createElement("div");
          row.className = "system-item";
          row.textContent = formatter(item);
          target.appendChild(row);
        });
      }

      function renderSystemManifest() {
        const manifest = SYSTEM_MANIFEST || {};
        const nodes = Array.isArray(manifest.nodes) ? manifest.nodes : [];
        const edges = Array.isArray(manifest.edges) ? manifest.edges : [];
        const tools = Array.isArray(manifest.tools) ? manifest.tools : [];
        const mcpServers = Array.isArray(manifest.mcp_servers) ? manifest.mcp_servers : [];
        const warnings = Array.isArray(manifest.warnings) ? manifest.warnings : [];

        systemNodeCount.textContent = String(nodes.length);
        systemToolCount.textContent = String(tools.length);
        systemMcpCount.textContent = String(mcpServers.length);
        renderList(systemNodes, nodes, "No system map available", (item) => item.name || item.id || "node");
        renderList(systemEdges, edges, "No system map available", (item) => (item.source || "?") + " -> " + (item.target || "?"));
        renderList(systemTools, tools, "No tools detected", (item) => (item.node_id ? item.node_id + ": " : "") + (item.name || "tool"));
        renderList(systemMcpServers, mcpServers, "No MCP servers detected", (item) => {
          const details = [item.name || "mcp_server", item.server_url, item.transport].filter(Boolean);
          return details.join(" · ");
        });
        systemMermaid.textContent = manifest.mermaid || "No system map available";
        renderList(systemWarnings, warnings, "No warnings", (item) => String(item));
      }

      function textFromPayload(payload) {
        if (!payload) return "";
        const value = payload.content || payload.message || payload.answer || payload.final || payload.output || payload.response || "";
        return typeof value === "string" ? value : JSON.stringify(value);
      }

      function agentLabel(payload) {
        if (!payload) return "Agent";
        return payload.agent_id || payload.agent || payload.agent_name || payload.name || payload.node_id || payload.node || "Agent";
      }

      function toolLabel(payload) {
        if (!payload) return "tool";
        return payload.tool_name || payload.tool || payload.name || "tool";
      }

      function addMessage(role, text, options = {}) {
        hideEmptyState();
        const row = document.createElement("div");
        row.className = "message-row " + role;
        const bubble = document.createElement("div");
        bubble.className = "message " + role + (options.final ? " final" : "");
        bubble.textContent = text;
        if (options.agent) {
          bubble.dataset.agent = options.agent;
        }
        row.appendChild(bubble);
        timelineInner.appendChild(row);
        scrollTimeline();
        return bubble;
      }

      function ensureAssistantMessage(payload) {
        if (!currentAssistantMessage) {
          currentAssistantMessage = addMessage("assistant", "", { agent: agentLabel(payload) });
        }
        return currentAssistantMessage;
      }

      function addCard(kind, title, body, badgeText) {
        hideEmptyState();
        const card = document.createElement("section");
        card.className = "event-card " + kind;
        const heading = document.createElement("div");
        heading.className = "card-title";
        const label = document.createElement("span");
        label.textContent = title;
        heading.appendChild(label);
        if (badgeText) {
          const badge = document.createElement("span");
          badge.className = "badge " + (kind === "tool" ? "tool" : kind === "node" ? "node" : "");
          badge.textContent = badgeText;
          heading.appendChild(badge);
        }
        card.appendChild(heading);
        if (body) {
          const detail = document.createElement("div");
          detail.className = "card-body";
          detail.textContent = truncateText(body, 700);
          card.appendChild(detail);
        }
        timelineInner.appendChild(card);
        scrollTimeline();
        return card;
      }

      function addReasoning(payload) {
        const text = payload.thought || payload.reasoning || payload.summary || payload.plan || "";
        if (typeof text !== "string" || !text) return;
        hideEmptyState();
        const details = document.createElement("details");
        details.className = "event-card agent reasoning";
        const summary = document.createElement("summary");
        summary.textContent = "Reasoning";
        const body = document.createElement("div");
        body.className = "card-body";
        body.textContent = "Observable reasoning emitted by the system. This is not hidden model reasoning.\\n\\n" + truncateText(text, 900);
        details.appendChild(summary);
        details.appendChild(body);
        timelineInner.appendChild(details);
        scrollTimeline();
      }

      function addHandoff(payload, eventName) {
        const fromAgent = payload.from_agent || payload.from || payload.source_agent || agentLabel(payload);
        const toAgent = payload.to_agent || payload.to || payload.target_agent || "Agent";
        hideEmptyState();
        const item = document.createElement("div");
        item.className = "handoff";
        item.textContent = fromAgent + " -> " + toAgent + " · " + eventName;
        timelineInner.appendChild(item);
        scrollTimeline();
      }

      function addToolStarted(payload) {
        const name = toolLabel(payload);
        const body = payload.input || payload.args || payload.arguments ? "Input: " + truncateText(JSON.stringify(payload.input || payload.args || payload.arguments), 500) : "";
        const card = addCard("tool", "Tool call", body, name);
        activeTools.set(name, card);
      }

      function addToolEnded(payload) {
        const name = toolLabel(payload);
        const card = activeTools.get(name) || addCard("tool", "Tool result", "", name);
        const status = payload.error || payload.status === "error" ? "error" : "success";
        const badge = card.querySelector(".badge");
        if (badge) {
          badge.className = "badge " + status;
          badge.textContent = status;
        }
        const output = payload.output || payload.result || payload.response || payload.error || "";
        if (output) {
          let body = card.querySelector(".card-body");
          if (!body) {
            body = document.createElement("div");
            body.className = "card-body";
            card.appendChild(body);
          }
          body.textContent = "Output: " + truncateText(typeof output === "string" ? output : JSON.stringify(output), 700);
        }
        activeTools.delete(name);
      }

      function openTelemetry() {
        telemetryDrawer.classList.add("open");
        telemetryDrawer.setAttribute("aria-hidden", "false");
        telemetryBackdrop.hidden = false;
      }

      function closeTelemetry() {
        telemetryDrawer.classList.remove("open");
        telemetryDrawer.setAttribute("aria-hidden", "true");
        telemetryBackdrop.hidden = true;
      }

      function openSystem() {
        systemDrawer.classList.add("open");
        systemDrawer.setAttribute("aria-hidden", "false");
        systemBackdrop.hidden = false;
      }

      function closeSystem() {
        systemDrawer.classList.remove("open");
        systemDrawer.setAttribute("aria-hidden", "true");
        systemBackdrop.hidden = true;
      }

      function estimateInputTokens(value) {
        if (typeof value === "string") {
          return countTokens(value);
        }
        if (Array.isArray(value)) {
          return value.reduce((acc, item) => acc + estimateInputTokens(item), 0);
        }
        if (value && typeof value === "object") {
          return Object.values(value).reduce((acc, item) => acc + estimateInputTokens(item), 0);
        }
        return 0;
      }

      function formatMs(value) {
        if (typeof value !== "number" || Number.isNaN(value)) return "-";
        return value.toFixed(1) + " ms";
      }

      function formatCost(value) {
        if (typeof value !== "number" || Number.isNaN(value)) return "not configured";
        return "$" + value.toFixed(4);
      }

      function currentNode() {
        return stack[stack.length - 1];
      }

      function pushNode(type, label) {
        const node = { type, label, children: [] };
        currentNode().children.push(node);
        stack.push(node);
      }

      function popNode() {
        if (stack.length > 1) {
          stack.pop();
        }
      }

      function addLeaf(type, label) {
        currentNode().children.push({ type, label });
      }

      function renderTree() {
        if (debugTree) {
          debugTree.textContent = JSON.stringify(tree, null, 2);
        }
      }

      function updateMetrics() {
        if (metricTotalEvents) metricTotalEvents.textContent = String(totalEvents);
        if (metricInputTokens) metricInputTokens.textContent = String(inputTokens);
        if (metricReasoningTokens) metricReasoningTokens.textContent = String(reasoningTokens);
        if (metricOutputTokens) metricOutputTokens.textContent = String(outputTokens);
        if (metricTotalTokens) metricTotalTokens.textContent = String(totalTokens);
        if (metricProvider) metricProvider.textContent = provider;
        if (metricModel) metricModel.textContent = model;
        if (metricCostUsd) metricCostUsd.textContent = formatCost(costUsd);
        if (metricLatency) metricLatency.textContent = formatMs(latencyMs);
        if (metricTtft) metricTtft.textContent = formatMs(ttftMs);
        if (metricErrorCount) metricErrorCount.textContent = String(errorCount);
        if (metricRetryCount) metricRetryCount.textContent = String(retryCount);
        if (metricEstimatedTokens) metricEstimatedTokens.textContent = estimatedTokens ? "true" : "false";
        if (metricEstimatedCost) metricEstimatedCost.textContent = estimatedCost ? "true" : "false";
      }

      function buildWebSocketUrl() {
        const protocol =
          window.location.protocol === "https:"
            ? "wss://"
            : "ws://";

        let path = window.location.pathname || "/";

        // eliminar slash final
        path = path.replace(/\/$/, "");

        // localhost
        if (path === "") {
          path = "";
        }

        return protocol +
              window.location.host +
              path +
              "/ws";
      }

      function resetRunMetrics() {
          runStartedAt = performance.now();
          firstTokenAt = null;
          currentAssistantMessage = null;
          sawTokenInCurrentRun = false;
      }

      function resetState() {
        token.textContent = "";
        events.textContent = "";
        if (debugEvents) debugEvents.textContent = "";
        if (debugTree) debugTree.textContent = "";
        timelineInner.textContent = "";
        if (emptyState) {
          emptyState.hidden = false;
          timelineInner.appendChild(emptyState);
        }

        totalEvents = 0;
        inputTokens = 0;
        outputTokens = 0;
        totalTokens = 0;
        provider = "-";
        model = "-";
        costUsd = null;
        latencyMs = null;
        ttftMs = null;
        errorCount = 0;
        retryCount = 0;
        estimatedTokens = true;
        estimatedCost = true;
        sawRealUsage = false;
        sawAgentToken = false;
        runStartedAt = performance.now();
        firstTokenAt = null;
        tree = [{ type: "session", label: "session", children: [] }];
        stack = [tree[0]];
        currentAssistantMessage = null;
        activeTools = new Map();
        sawTokenInCurrentRun = false;

        updateMetrics();
        renderTree();
      }

      function readProvider(payload) {
        // payload.metadata.ls_provider payload.metadata.provider
        if (payload && typeof payload.provider === "string" && payload.provider) {
          return payload.provider;
        }
        if (payload && payload.metadata && typeof payload.metadata.ls_provider === "string" && payload.metadata.ls_provider) {
          return payload.metadata.ls_provider;
        }
        if (payload && payload.metadata && typeof payload.metadata.provider === "string" && payload.metadata.provider) {
          return payload.metadata.provider;
        }
        return null;
      }

      function readModel(payload) {
        // payload.response_metadata.model_name payload.response_metadata.model
        if (payload && typeof payload.model === "string" && payload.model) {
          return payload.model;
        }
        if (payload && payload.metadata && typeof payload.metadata.ls_model_name === "string" && payload.metadata.ls_model_name) {
          return payload.metadata.ls_model_name;
        }
        if (payload && payload.response_metadata && typeof payload.response_metadata.model_name === "string" && payload.response_metadata.model_name) {
          return payload.response_metadata.model_name;
        }
        if (payload && payload.response_metadata && typeof payload.response_metadata.model === "string" && payload.response_metadata.model) {
          return payload.response_metadata.model;
        }
        return null;
      }

      function readUsage(payload) {
        // payload.usage.input_tokens payload.usage.output_tokens payload.usage.total_tokens
        const usage = payload && payload.usage && typeof payload.usage === "object" ? payload.usage : null;
        const input = usage && typeof usage.input_tokens === "number"
          ? usage.input_tokens
          : (typeof payload.input_tokens === "number"
            ? payload.input_tokens
            : (typeof payload.prompt_tokens === "number"
              ? payload.prompt_tokens
              : (typeof payload.tokens_in === "number" ? payload.tokens_in : null)));
        const output = usage && typeof usage.output_tokens === "number"
          ? usage.output_tokens
          : (typeof payload.output_tokens === "number"
            ? payload.output_tokens
            : (typeof payload.completion_tokens === "number"
              ? payload.completion_tokens
              : (typeof payload.tokens_out === "number" ? payload.tokens_out : null)));
        const total = usage && typeof usage.total_tokens === "number"
          ? usage.total_tokens
          : (typeof payload.total_tokens === "number" ? payload.total_tokens : null);
        return { usage, input, output, total };
      }

      function applyRealMetrics(payload) {
        const usageInfo = readUsage(payload);
        const hasUsage = Boolean(usageInfo.usage || usageInfo.input !== null || usageInfo.output !== null || usageInfo.total !== null);

        if (hasUsage) {

          if (usageInfo.input !== null) {
            inputTokens = usageInfo.input;
          }

          if (usageInfo.output !== null) {
            outputTokens = usageInfo.output;
          }

          if (
            usageInfo.usage &&
            usageInfo.usage.output_token_details &&
            typeof usageInfo.usage.output_token_details.reasoning === "number"
          ) {
            reasoningTokens =
              usageInfo.usage.output_token_details.reasoning;

            outputTokens = Math.max(
              outputTokens - reasoningTokens,
              0
            );
          }

          if (usageInfo.total !== null) {
            totalTokens = usageInfo.total;
          } else if (
            usageInfo.input !== null ||
            usageInfo.output !== null
          ) {
            totalTokens =
              inputTokens +
              outputTokens +
              reasoningTokens;
          }

          estimatedTokens = false;
          sawRealUsage = true;
        }

        const providerValue = readProvider(payload);
        if (providerValue !== null) {
          provider = providerValue;
        }

        const modelValue = readModel(payload);
        if (modelValue !== null) {
          model = modelValue;
        }

        const costValue = payload && payload.cost && typeof payload.cost.cost_usd === "number"
          ? payload.cost.cost_usd
          : (typeof payload.cost_usd === "number" ? payload.cost_usd : null);
        if (costValue !== null) {
          costUsd = costValue;
          estimatedCost = false;
        }

        const timing = payload && payload.timing && typeof payload.timing === "object" ? payload.timing : null;
        if (timing && typeof timing.latency_ms === "number") {
          latencyMs = timing.latency_ms;
        }
        if (timing && typeof timing.ttft_ms === "number") {
          ttftMs = timing.ttft_ms;
        }

        if (typeof payload.total_events === "number") {
          totalEvents = payload.total_events;
        }
        if (typeof payload.error_count === "number") {
          errorCount = payload.error_count;
        }
        if (typeof payload.retry_count === "number") {
          retryCount = payload.retry_count;
        }
        if (payload.estimated && typeof payload.estimated.tokens === "boolean") {
          estimatedTokens = payload.estimated.tokens;
          sawRealUsage = sawRealUsage || payload.estimated.tokens === false;
        }
        if (payload.estimated && typeof payload.estimated.cost === "boolean") {
          estimatedCost = payload.estimated.cost;
        }
      }

      function handleAgentToken(event) {
        const payload = event.payload || {};
        const delta = payload.delta || payload.token || "";
        if (typeof delta !== "string" || !delta) return;
        const bubble = ensureAssistantMessage(payload);
        bubble.textContent += delta;
        token.textContent += delta;
        if (!sawRealUsage) {
          outputTokens += countTokens(delta);
          totalTokens = inputTokens + outputTokens;
          estimatedTokens = true;
        }
        if (firstTokenAt === null) {
          firstTokenAt = performance.now();
        }
        sawAgentToken = true;
        sawTokenInCurrentRun = true;
      }

      function handleAgentMessage(event) {
        const payload = event.payload || {};
        if (sawTokenInCurrentRun) {
          return;
        }
        const content = textFromPayload(payload);
        if (!content) return;
        addMessage("assistant", content, { agent: agentLabel(payload), final: true });
        currentAssistantMessage = null;
        if (!sawRealUsage && !sawAgentToken) {
          outputTokens += countTokens(content);
          totalTokens = inputTokens + outputTokens;
          estimatedTokens = true;
        }
      }

      function handleSessionEnded(event) {
        const payload = event.payload || {};
        const finalText = payload.answer || payload.final || payload.output;
        if (typeof finalText === "string" && finalText) {
          addMessage("assistant", finalText, { final: true });
        }
      }

      function updateTreeFromEvent(event) {
        const payload = event.payload || {};

        if (event.name === "node.started") {
          pushNode("node.started", payload.name || payload.node || payload.id || "node");
          return;
        }

        if (event.name === "llm.started") {
          const labelParts = [];
          const providerValue = readProvider(payload);
          const modelValue = readModel(payload);
          if (providerValue) labelParts.push(providerValue);
          if (modelValue) labelParts.push(modelValue);
          pushNode("llm.started", labelParts.length > 0 ? labelParts.join(" / ") : "llm.started");
          return;
        }

        if (event.name === "tool.started") {
          pushNode("tool.started", payload.tool || payload.name || "tool");
          return;
        }

        if (event.name === "agent.message") {
          const content = payload.content || payload.message || "";
          addLeaf("agent.message", truncateText(typeof content === "string" ? content : "", 80));
          return;
        }

        if (event.name === "node.ended" || event.name === "llm.ended" || event.name === "tool.ended") {
          popNode();
        }
      }

      function handleEvent(event) {
        totalEvents += 1;

        const payload = event.payload || {};

        if (event.name === "agent.token") handleAgentToken(event);
        if (event.name === "agent.message") handleAgentMessage(event);
        if (event.name === "agent.thinking") addReasoning(payload);
        if (event.name === "agent.started") addCard("agent", "Agent started", "", agentLabel(payload));
        if (event.name === "agent.ended") addCard("agent", "Agent ended", "", agentLabel(payload));
        const isHandoffEvent = event.name === "agent.call" || event.name === "agent.handoff" || event.name === "agent.delegate";
        if (isHandoffEvent) addHandoff(payload, event.name);
        if (!isHandoffEvent && (payload.to_agent || payload.from_agent)) addHandoff(payload, event.name);
        if (event.name === "tool.started") addToolStarted(payload);
        if (event.name === "tool.ended") addToolEnded(payload);
        if (event.name === "session.ended") handleSessionEnded(event);
        if (event.name === "error") addCard("error", "Error", payload.message || "Unknown error", payload.type || "error");

        if (event.name === "metrics.updated" || event.name === "llm.started" || event.name === "llm.ended") {
          applyRealMetrics(payload);
        }

        if (event.name === "error") {
          errorCount += 1;
        }

        if (event.name && event.name.toLowerCase().includes("retry")) {
          retryCount += 1;
        }

        updateTreeFromEvent(event);
        updateMetrics();
        renderTree();
      }

      function handleRunClick() {
        resetRunMetrics();
        appendEvent("Run clicked");

        const promptText = input.value.trim();
        if (!promptText) {
          return;
        }
        const parsed = {
          prompt: promptText,
          conversation_id: conversationId
        };
        addMessage("user", promptText);
        input.value = "";
        input.focus();      
        inputTokens = estimateInputTokens(parsed);
        totalTokens = inputTokens;
        updateMetrics();

        const wsUrl = buildWebSocketUrl();
        appendEvent("Opening WebSocket: " + wsUrl);

        socket = new WebSocket(wsUrl);

        socket.onopen = function () {
          appendEvent("WebSocket opened");
          socket.send(JSON.stringify(parsed));
        };

        socket.onmessage = function (message) {
          const event = JSON.parse(message.data);
          appendEvent(JSON.stringify(event));
          handleEvent(event);
        };

        socket.onerror = function () {
          appendEvent("WebSocket error");
        };

        socket.onclose = function () {
          appendEvent("WebSocket closed");
        };
      }

      if (!runButton) {
        appendEvent("client.error: missing run button");
        return;
      }

      composer.addEventListener("submit", function (submitEvent) {
        submitEvent.preventDefault();
        handleRunClick();
      });
      input.addEventListener("keydown", function (keyboardEvent) {
        if (keyboardEvent.key === "Enter" && !keyboardEvent.shiftKey) {
          keyboardEvent.preventDefault();
          handleRunClick();
        }
      });
      telemetryToggle.addEventListener("click", openTelemetry);
      telemetryClose.addEventListener("click", closeTelemetry);
      telemetryBackdrop.addEventListener("click", closeTelemetry);
      systemToggle.addEventListener("click", openSystem);
      systemClose.addEventListener("click", closeSystem);
      systemBackdrop.addEventListener("click", closeSystem);
      renderSystemManifest();
      updateMetrics();
      renderTree();
    });
  </script>
</body>
</html>"""
        return html.replace("__DEBUG_PANEL__", debug_panel).replace("__SYSTEM_MANIFEST__", system_manifest_json)
