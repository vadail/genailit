from __future__ import annotations

from collections.abc import AsyncIterator, Iterable

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from genailit import AdapterContext, BaseAgentAdapter, GenAILitApp, GenAILitEvent
from genailit.core import _websocket_path_for_pathname


def test_genailit_app_can_be_created() -> None:
    class DemoAdapter(BaseAgentAdapter):
        def build_events(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
            return ()

    app = GenAILitApp(adapter=DemoAdapter())

    assert app.adapter.__class__.__name__ == "DemoAdapter"
    assert app.telemetry.snapshot() == ()
    assert app.asgi_app.__class__.__name__ == "FastAPI"


def test_genailit_app_can_be_created_without_adapter() -> None:
    app = GenAILitApp()

    assert app.adapter is None
    assert app.telemetry.snapshot() == ()
    assert app.asgi_app.__class__.__name__ == "FastAPI"


def test_health_endpoint() -> None:
    class DemoAdapter(BaseAgentAdapter):
        def build_events(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
            return ()

    client = TestClient(GenAILitApp(adapter=DemoAdapter()).asgi_app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_returns_minimal_html() -> None:
    class DemoAdapter(BaseAgentAdapter):
        def build_events(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
            return ()

    client = TestClient(GenAILitApp(adapter=DemoAdapter()).asgi_app)

    response = client.get("/")

    assert response.status_code == 200
    body = response.text
    assert 'id="timeline"' in body
    assert 'class="chat-timeline"' in body
    assert 'id="composer"' in body
    assert 'id="run"' in body
    assert 'id="input"' in body
    assert 'placeholder="Escribe tu mensaje..."' in body
    assert ">Send</button>" in body
    assert 'id="system-toggle"' in body
    assert 'id="system-drawer"' in body
    assert "System Map" in body
    assert "No system map available" in body
    assert 'id="events"' in body
    assert 'id="token"' in body
    assert 'id="telemetry-drawer"' in body
    assert 'aria-hidden="true"' in body
    assert 'id="raw-events-details"' in body
    assert '<details id="raw-events-details">' in body
    assert 'id="metric-provider"' in body
    assert 'id="metric-model"' in body
    assert 'id="metric-cost-usd"' in body
    assert 'id="metric-estimated-tokens"' in body
    assert 'id="metric-estimated-cost"' in body
    assert "Run clicked" in body
    assert "Opening WebSocket" in body
    assert "new WebSocket" in body
    assert "payload.usage.input_tokens" in body
    assert "payload.metadata.ls_provider" in body
    assert "payload.response_metadata.model_name" in body
    assert "handleAgentToken" in body
    assert "agent.thinking" in body
    assert "agent.call" in body
    assert "agent.handoff" in body
    assert "agent.delegate" in body
    assert "tool.started" in body
    assert "tool.ended" in body
    assert "handleSessionEnded" in body
    assert "const parsed = { prompt: promptText };" in body
    assert 'keyboardEvent.key === "Enter" && !keyboardEvent.shiftKey' in body
    assert "JSON.parse(input.value" not in body
    assert "window.location.pathname" in body
    assert "basePath === \"/\" ? \"/ws\"" in body
    assert "basePath.replace(/\\/$/, \"\") + \"/ws\"" in body
    assert "LLM started" not in body
    assert "LLM ended" not in body
    assert "Node started" not in body
    assert "Node ended" not in body
    assert "if (sawTokenInCurrentRun)" in body
    assert "not configured" in body


def test_websocket_path_resolution_supports_local_and_sagemaker_paths() -> None:
    assert _websocket_path_for_pathname("/") == "/ws"
    assert _websocket_path_for_pathname("/proxy/8501/") == "/proxy/8501/ws"


def test_websocket_streams_events_and_records_telemetry() -> None:
    calls: list[tuple[dict[str, object], AdapterContext]] = []

    class FakeAdapter(BaseAgentAdapter):
        def build_events(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
            return ()

        def stream(self, input_data: object, context: AdapterContext) -> Iterable[GenAILitEvent]:
            calls.append((input_data, context))
            yield GenAILitEvent(name="agent.token", payload={"token": "Hel"})
            yield GenAILitEvent(name="agent.token", payload={"token": "lo"})
            yield GenAILitEvent(name="agent.done", payload={"ok": True})

    app = GenAILitApp(adapter=FakeAdapter())
    client = TestClient(app.asgi_app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"prompt": "Hi"})
        first = websocket.receive_json()
        second = websocket.receive_json()
        third = websocket.receive_json()

    assert first == {"name": "agent.token", "payload": {"token": "Hel"}}
    assert second == {"name": "agent.token", "payload": {"token": "lo"}}
    assert third == {"name": "agent.done", "payload": {"ok": True}}
    assert app.telemetry.snapshot() == (
        GenAILitEvent(name="agent.token", payload={"token": "Hel"}),
        GenAILitEvent(name="agent.token", payload={"token": "lo"}),
        GenAILitEvent(name="agent.done", payload={"ok": True}),
    )
    assert calls[0][0] == {"prompt": "Hi"}
    assert calls[0][1].session_id
    assert calls[0][1].run_id
    assert calls[0][1].input_data == {"prompt": "Hi"}


def test_websocket_captures_adapter_errors() -> None:
    class ErrorAdapter(BaseAgentAdapter):
        def build_events(self, context: AdapterContext) -> Iterable[GenAILitEvent]:
            return ()

        def stream(self, input_data: object, context: AdapterContext) -> Iterable[GenAILitEvent]:
            raise RuntimeError("boom")

    client = TestClient(GenAILitApp(adapter=ErrorAdapter()).asgi_app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"prompt": "Hi"})
        event = websocket.receive_json()
        assert event["name"] == "error"
        assert event["payload"]["message"] == "boom"
        assert event["payload"]["type"] == "RuntimeError"
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()


def test_agent_decorator_streams_async_strings() -> None:
    app = GenAILitApp()

    @app.agent
    async def demo(input_data: object, context: AdapterContext):
        yield "Hola "
        yield "desde "
        yield "GenAILit"

    client = TestClient(app.asgi_app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"prompt": "Hi"})
        events = [websocket.receive_json() for _ in range(5)]

    assert [event["name"] for event in events] == [
        "session.started",
        "agent.token",
        "agent.token",
        "agent.token",
        "session.ended",
    ]
    assert "".join(event["payload"].get("delta", "") for event in events) == "Hola desde GenAILit"


def test_agent_decorator_sync_function_can_return_string() -> None:
    app = GenAILitApp()

    @app.agent
    def demo(input_data: object, context: AdapterContext) -> str:
        return "Hola sync"

    client = TestClient(app.asgi_app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"prompt": "Hi"})
        events = [websocket.receive_json() for _ in range(3)]

    assert [event["name"] for event in events] == [
        "session.started",
        "agent.token",
        "session.ended",
    ]
    assert events[1]["payload"] == {"delta": "Hola sync"}


def test_agent_decorator_sync_function_can_return_iterable() -> None:
    app = GenAILitApp()

    @app.agent
    def demo(input_data: object, context: AdapterContext) -> list[str]:
        return ["Hola ", "iterable"]

    client = TestClient(app.asgi_app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"prompt": "Hi"})
        events = [websocket.receive_json() for _ in range(4)]

    assert [event["name"] for event in events] == [
        "session.started",
        "agent.token",
        "agent.token",
        "session.ended",
    ]
    assert "".join(event["payload"].get("delta", "") for event in events) == "Hola iterable"


def test_agent_decorator_async_function_can_return_async_iterable() -> None:
    app = GenAILitApp()

    async def pieces() -> AsyncIterator[str]:
        yield "Hola "
        yield "async iterable"

    @app.agent
    async def demo(input_data: object, context: AdapterContext) -> AsyncIterator[str]:
        return pieces()

    client = TestClient(app.asgi_app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"prompt": "Hi"})
        events = [websocket.receive_json() for _ in range(4)]

    assert [event["name"] for event in events] == [
        "session.started",
        "agent.token",
        "agent.token",
        "session.ended",
    ]
    assert "".join(event["payload"].get("delta", "") for event in events) == "Hola async iterable"


def test_agent_decorator_can_return_genailit_event() -> None:
    app = GenAILitApp()

    @app.agent
    def demo(input_data: object, context: AdapterContext) -> GenAILitEvent:
        return GenAILitEvent(name="agent.message", payload={"content": "Hola"})

    client = TestClient(app.asgi_app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"prompt": "Hi"})
        first = websocket.receive_json()
        second = websocket.receive_json()
        third = websocket.receive_json()

    assert first["name"] == "session.started"
    assert second == {"name": "agent.message", "payload": {"content": "Hola"}}
    assert third["name"] == "session.ended"


def test_websocket_without_agent_or_adapter_returns_clear_error() -> None:
    client = TestClient(GenAILitApp().asgi_app)

    with client.websocket_connect("/ws") as websocket:
        websocket.send_json({"prompt": "Hi"})
        event = websocket.receive_json()
        assert event["name"] == "error"
        assert "Use @app.agent or pass adapter=..." in event["payload"]["message"]
        with pytest.raises(WebSocketDisconnect):
            websocket.receive_json()
