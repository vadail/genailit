from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from genailit import AdapterContext, GenAILitEvent


def test_demo_adapter_streams_expected_events() -> None:
    module_path = Path(__file__).resolve().parents[1] / "examples" / "simple_demo.py"
    spec = spec_from_file_location("simple_demo", module_path)
    assert spec is not None
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    demo_adapter = module.DemoAdapter()
    context = AdapterContext(session_id="demo")

    events = tuple(demo_adapter.stream({}, context))

    assert events == (
        GenAILitEvent(name="session.started"),
        GenAILitEvent(name="agent.thinking"),
        GenAILitEvent(name="agent.token", payload={"token": "Hola "}),
        GenAILitEvent(name="agent.token", payload={"token": "desde "}),
        GenAILitEvent(name="agent.token", payload={"token": "GenAILit"}),
        GenAILitEvent(name="tool.started"),
        GenAILitEvent(name="tool.ended"),
        GenAILitEvent(name="metrics.updated"),
        GenAILitEvent(name="session.ended"),
    )
