from __future__ import annotations

from genailit.debug import DebugPanel


def test_debug_panel_exposes_primary_dom_ids() -> None:
    html = DebugPanel().render()

    assert 'id="telemetry-drawer"' in html
    assert 'aria-hidden="true"' in html
    assert 'id="telemetry-backdrop"' in html
    assert 'id="telemetry-close"' in html
    assert '<details id="raw-events-details">' in html
    assert 'id="system-drawer"' in html
    assert 'aria-label="System Map"' in html
    assert 'id="system-backdrop"' in html
    assert 'id="system-close"' in html
    assert "No system map available" in html

    for element_id in (
        "metric-total-events",
        "metric-input-tokens",
        "metric-output-tokens",
        "metric-total-tokens",
        "metric-provider",
        "metric-model",
        "metric-cost-usd",
        "metric-latency",
        "metric-ttft",
        "metric-error-count",
        "metric-retry-count",
        "metric-estimated-tokens",
        "metric-estimated-cost",
        "debug-tree",
        "debug-events",
        "system-node-count",
        "system-tool-count",
        "system-mcp-count",
        "system-nodes",
        "system-edges",
        "system-tools",
        "system-mcp-servers",
        "system-mermaid",
        "system-warnings",
    ):
        assert f'id="{element_id}"' in html
