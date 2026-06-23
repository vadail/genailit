from __future__ import annotations


class DebugPanel:
    def render(self) -> str:
        return """
    <div id="telemetry-backdrop" class="telemetry-backdrop" hidden></div>
    <aside id="telemetry-drawer" class="telemetry-drawer" aria-label="Telemetry" aria-hidden="true">
      <div class="drawer-header">
        <div>
          <p class="eyebrow">Observability</p>
          <h2>Telemetry</h2>
        </div>
        <button id="telemetry-close" class="ghost-button" type="button">Close</button>
      </div>
      <section>
        <h3>Metrics</h3>
        <dl class="metrics-grid">
          <dt>Total events</dt><dd id="metric-total-events">0</dd>
          <dt>Input tokens</dt><dd id="metric-input-tokens">0</dd>
          <dt>Visible output tokens</dt><dd id="metric-output-tokens">0</dd>
          <dt>Reasoning tokens</dt>
          <dd id="metric-reasoning-tokens">0</dd>
          <dt>Total billed tokens</dt>
          <dd id="metric-total-tokens">0</dd>
          <dt>Provider</dt><dd id="metric-provider">-</dd>
          <dt>Model</dt><dd id="metric-model">-</dd>
          <dt>Cost USD</dt><dd id="metric-cost-usd">-</dd>
          <dt>Latency</dt><dd id="metric-latency">-</dd>
          <dt>TTFT</dt><dd id="metric-ttft">-</dd>
          <dt>Error count</dt><dd id="metric-error-count">0</dd>
          <dt>Retry count</dt><dd id="metric-retry-count">0</dd>
          <dt>Estimated tokens</dt><dd><span class="badge" id="metric-estimated-tokens">true</span></dd>
          <dt>Estimated cost</dt><dd><span class="badge" id="metric-estimated-cost">true</span></dd>
        </dl>
      </section>
      <section>
        <h3>Execution tree</h3>
        <pre id="debug-tree"></pre>
      </section>
      <details id="raw-events-details">
        <summary>Raw events</summary>
        <pre id="debug-events"></pre>
      </details>
    </aside>
    <div id="system-backdrop" class="telemetry-backdrop" hidden></div>
    <aside id="system-drawer" class="telemetry-drawer" aria-label="System Map" aria-hidden="true">
      <div class="drawer-header">
        <div>
          <p class="eyebrow">System</p>
          <h2>System Map</h2>
        </div>
        <button id="system-close" class="ghost-button" type="button">Close</button>
      </div>
      <section>
        <h3>Summary</h3>
        <dl class="metrics-grid">
          <dt>Agents / nodes</dt><dd id="system-node-count">0</dd>
          <dt>Tools</dt><dd id="system-tool-count">0</dd>
          <dt>MCP servers</dt><dd id="system-mcp-count">0</dd>
        </dl>
      </section>
      <section>
        <h3>Nodes</h3>
        <div id="system-nodes" class="system-list">No system map available</div>
      </section>
      <section>
        <h3>Edges</h3>
        <div id="system-edges" class="system-list">No system map available</div>
      </section>
      <section>
        <h3>Tools</h3>
        <div id="system-tools" class="system-list">No tools detected</div>
      </section>
      <section>
        <h3>MCP servers</h3>
        <div id="system-mcp-servers" class="system-list">No MCP servers detected</div>
      </section>
      <details>
        <summary>Graph text</summary>
        <pre id="system-mermaid">No system map available</pre>
      </details>
      <section>
        <h3>Warnings</h3>
        <div id="system-warnings" class="system-list">No warnings</div>
      </section>
    </aside>
"""
