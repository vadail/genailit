# GenAILit

GenAILit is a small Python library for building web interfaces for multi-agent systems.
It gives you a single-process, single-port runtime with FastAPI, WebSocket streaming, and
an embedded DebugPanel for LLMOps-style inspection.

## What it is

GenAILit is not just a chat UI.
It is a lightweight runtime and adapter layer for agentic applications where you want to:

- expose an agent over the web from pure Python
- stream events and tokens over WebSocket
- inspect execution traces and metrics
- stay compatible with SageMaker Studio and other proxy-based environments

## Why it is not only a chat UI

A chat UI only renders messages.
GenAILit also gives you:

- a framework-agnostic event model
- telemetry for tokens, latency, provider, model, cost, errors, and retries
- adapters for different backends
- a built-in debug surface for tracing multi-agent execution

That makes it useful for building and auditing multi-agent systems, not only for chatting with them.

## Installation

```bash
pip install genailit
```

Optional LangGraph support:

```bash
pip install "genailit[langgraph]"
```

Requirements:

- Python 3.10+
- `pydantic>=2`
- `fastapi>=0.110`
- `uvicorn[standard]>=0.27`

## Quickstart

The simplest way to build an app is with `@app.agent`:

```python
from genailit import GenAILitApp

app = GenAILitApp()


@app.agent
async def demo(input_data, context):
    yield "Hola "
    yield "desde "
    yield "GenAILit"


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
```

Run it with the CLI:

```bash
genailit run app.py --host 0.0.0.0 --port 8501
```

The UI opens a WebSocket to the same host and port, so it works well behind SageMaker and similar proxies.

## Using LangGraph

LangGraph is optional and stays outside the core runtime.
Install the extra first:

```bash
pip install "genailit[langgraph]"
```

Then use the adapter:

```python
from genailit import GenAILitApp
from genailit.adapters.langgraph import LangGraphAdapter

graph = ...
adapter = LangGraphAdapter(graph)
app = GenAILitApp(adapter=adapter)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
```

See [examples/langgraph_demo.py](examples/langgraph_demo.py) for a minimal runnable example.

## Examples

- [examples/function_agent.py](examples/function_agent.py) - simplest `@app.agent` demo
- [examples/langgraph_demo.py](examples/langgraph_demo.py) - minimal LangGraph integration
- [examples/sagemaker_quickstart.py](examples/sagemaker_quickstart.py) - SageMaker-friendly quickstart

## Architecture

GenAILit is intentionally split into small pieces:

- `core` - the FastAPI runtime and WebSocket server
- `adapters` - bridge layers for agent backends
- `events` - the framework-agnostic `GenAILitEvent` contract
- `telemetry` - in-memory session traces and metrics
- `DebugPanel` - the built-in execution inspector

The core stays agnostic.
Adapters translate backend-specific behavior into GenAILit events.

## Event catalog

Every streamed item uses the same shape:

```python
GenAILitEvent(name="event.name", payload={...})
```

Canonical event names for `0.1.x`:

| Event | Recommended payload |
| --- | --- |
| `session.started` | `{"session_id": str, "run_id": str \| None}` |
| `session.ended` | `{"session_id": str, "run_id": str \| None}` |
| `agent.token` | `{"delta": str}`. Legacy `{"token": str}` is still accepted by the UI and telemetry. |
| `agent.message` | `{"content": str}`. Legacy `{"message": str}` is still accepted. |
| `node.started` | `{"name": str}` or `{"session_id": str, "run_id": str \| None}` for root execution nodes. |
| `node.ended` | `{"name": str}` or `{"run_id": str \| None}` for root execution nodes. |
| `tool.started` | `{"name": str}` or `{"tool_name": str}`. |
| `tool.ended` | `{"name": str}` or `{"tool_name": str}`. |
| `llm.started` | `{"provider": str, "model": str, "metadata": {"source": str}, "timing": {"latency_ms": None, "ttft_ms": None}}`. |
| `llm.ended` | `{"provider": str, "model": str, "usage": {...}, "metadata": {"source": str}, "timing": {...}}`. |
| `metrics.updated` | `{"usage": {"input_tokens": int, "output_tokens": int, "total_tokens": int}, "provider": str, "model": str, "timing": {"latency_ms": float, "ttft_ms": float \| None}, "cost": {"cost_usd": float}}`. |
| `error` | `{"message": str, "type": str}`. |

Payloads may include fewer fields when the backend cannot provide them.
Raw backend payloads are not included unless an adapter explicitly enables `include_raw`.

## SageMaker Studio

GenAILit is designed to work in SageMaker Studio with:

- single-process execution
- single-port serving
- no Node or Vite runtime
- host `0.0.0.0`
- port `8501`

That keeps the app simple to install with `pip` and avoids frontend build steps in runtime.

## Telemetry and LLMOps

GenAILit tracks observability data such as:

- input tokens
- output tokens
- total tokens
- provider
- model
- latency
- TTFT
- cost
- error count
- retry count

The DebugPanel surfaces those metrics alongside execution events and a basic execution tree.
When real usage metadata is available, GenAILit prefers it over token estimates.

`TelemetryStore.get_session_metrics(session_id)` returns:

```python
{
    "total_events": int,
    "input_tokens": int,
    "output_tokens": int,
    "total_tokens": int,
    "model": str | None,
    "provider": str | None,
    "latency_ms": float | None,
    "ttft_ms": float | None,
    "cost_usd": float | None,
    "error_count": int,
    "retry_count": int,
    "estimated": {"tokens": bool, "cost": bool},
}
```

Telemetry precedence:

1. Real usage wins: `payload.usage.input_tokens`, `payload.usage.output_tokens`, and `payload.usage.total_tokens`.
2. Legacy aliases are used next: `prompt_tokens`, `completion_tokens`, `tokens_in`, and `tokens_out`.
3. If no token metadata exists, output tokens are estimated from `agent.token` or `agent.message` text.

Provider and model are resolved from `payload.provider`, `payload.model`, `payload.metadata.ls_provider`, `payload.metadata.ls_model_name`, and `payload.response_metadata.model_name`.
Cost is never estimated; `cost_usd` is only populated from explicit `payload.cost.cost_usd` or `payload.cost_usd`.
Latency and TTFT prefer explicit `payload.timing` values and otherwise fall back to in-memory session timestamps.

## Public API stability

For `0.1.x`, the stable surface is:

- `GenAILitEvent(name, payload)`
- `GenAILitApp(adapter=None)`, `@app.agent`, `app.asgi_app`, and `app.run(...)`
- `AdapterContext` and `BaseAgentAdapter.stream(input_data, context)`
- `TelemetryStore.record`, `extend`, `snapshot`, `clear`, `get_session_trace`, and `get_session_metrics`
- `LangGraphAdapter(graph, input_key="messages", stream_mode=None, include_raw=False)`
- `genailit run app.py --host 0.0.0.0 --port 8501`

## Privacy defaults

GenAILit avoids storing sensitive content by default:

- raw payloads are not persisted unless explicitly requested
- prompts are not stored in new telemetry structures by default
- message bodies and token text are only shown where needed for the live UI

This keeps the default footprint small while still supporting inspection when you enable it.

## Status

GenAILit is experimental.
The public API may change as the library grows.
The current goal is to keep the runtime small, stable in SageMaker, and easy to reason about.
