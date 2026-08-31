// Objective 1: capacity-aware workload planning. This view only displays the
// two schedules selected by the paper's Queryosity schedulers and loaded from
// prepared artifacts. No scheduler search runs in the request path.

import { Api } from "./api.js";
import { int, pct } from "./format.js";

const METHODS = ["sweep_D", "sweep_beam_D"];

export class SchedulerView {
  constructor({ mount, workloads, presets, onExplore, onExplain }) {
    this.mount = mount;
    this.workloads = workloads;
    this.presets = presets;
    this.onExplore = onExplore;
    this.onExplain = onExplain;
    this.workload = workloads[0]?.name || null;
    this.capacity = presets[0]?.pages || 102400;
    this._token = 0;
    this._build();
  }

  _build() {
    this.mount.innerHTML = `
      <div class="section-title objective-head">
        <div><span class="objective-kicker">Objective 1</span><h2>Capacity-Aware Workload Planning</h2></div>
        <p>Select the same recurring workload under different buffer capacities and inspect the
        complete schedules retained by Sweep-D and Sweep+Beam-D. Each card reports the
        complete-order cache-simulation result used to select that schedule.</p>
      </div>
      <div class="planning-controls panel panel--tight" id="planning-controls"></div>
      <div id="planning-context" class="planning-context"></div>
      <div id="planning-body"></div>`;

    const controls = this.mount.querySelector("#planning-controls");
    this._wl = document.createElement("select");
    for (const w of this.workloads) this._wl.append(option(w.name, w.label));
    this._wl.value = this.workload;
    this._wl.addEventListener("change", () => { this.workload = this._wl.value; this.refresh(); });

    this._cap = document.createElement("select");
    for (const p of this.presets) this._cap.append(option(String(p.pages), `${int(p.pages)} pages · ${p.human}`));
    this._cap.value = String(this.capacity);
    this._cap.addEventListener("change", () => { this.capacity = parseInt(this._cap.value, 10); this.refresh(); });

    controls.append(field("Recurring workload", this._wl), field("Selected buffer capacity C", this._cap));
    this._context = this.mount.querySelector("#planning-context");
    this._body = this.mount.querySelector("#planning-body");
    this.refresh();
  }

  setContext(workload, cap) {
    if (this.workloads.some((w) => w.name === workload)) {
      this.workload = workload; this._wl.value = workload;
    }
    if (this.presets.some((p) => p.pages === cap)) {
      this.capacity = cap; this._cap.value = String(cap);
    }
    this.refresh();
  }

  async refresh() {
    const token = ++this._token;
    this._body.innerHTML = `<div class="notice">Loading selected Queryosity schedules…</div>`;
    const w = this.workloads.find((x) => x.name === this.workload);
    const p = this.presets.find((x) => x.pages === this.capacity);
    this._context.innerHTML = `<span>${escapeHtml(w?.label || this.workload)}</span><span class="context-arrow">+</span><span>${escapeHtml(p?.human || "")}</span><span class="context-arrow">→</span><strong>selected schedule may change with C</strong>`;
    try {
      const data = await Api.generated(this.workload, this.capacity);
      if (token !== this._token) return;
      this._render(data);
    } catch (err) {
      if (token !== this._token) return;
      this._body.innerHTML = `<div class="notice ${err.status === 404 ? "" : "err-inline"}">
        ${err.status === 404
          ? `Prepared research schedules are not available yet for this configuration. Run <code>python scripts/generate_demo_artifacts.py --all</code> on the Queryosity VM.`
          : `Could not load generated schedules: ${escapeHtml(err.message)}`}
      </div>`;
    }
  }

  _render(data) {
    const methods = data.methods || {};
    const cards = METHODS.filter((m) => methods[m]).map((m) => this._card(m, methods[m])).join("");
    if (!cards) {
      this._body.innerHTML = `<div class="notice">No Sweep-D or Sweep+Beam-D artifact is available for this selection.</div>`;
      return;
    }
    const mock = data.backend === "mock"
      ? `<div class="callout warn demo-warning">MOCK mode: the schedules below are illustrative development data, not paper-valid benchmark results.</div>` : "";
    this._body.innerHTML = `${mock}<div class="planning-grid">${cards}</div>
      <div class="score-semantics panel panel--tight">
        <strong>Complete-order score</strong>
        <span>F<sub>hit</sub> is computed by replaying the completed order through the shared page-level clock-sweep simulator. Directional reuse and regret guide construction; complete-order simulation selects the retained candidate.</span>
      </div>`;

    this._body.querySelectorAll("[data-explore]").forEach((b) => b.addEventListener("click", () => {
      const method = b.getAttribute("data-explore"); const e = methods[method];
      this.onExplore(this.workload, this.capacity, e.order, e.label, method);
    }));
    this._body.querySelectorAll("[data-explain]").forEach((b) => b.addEventListener("click", () => {
      this.onExplain(this.workload, this.capacity, b.getAttribute("data-explain"));
    }));
  }

  _card(method, entry) {
    const c = entry.construction || {};
    const wr = c.selected_w_regret;
    const beam = c.beam_width;
    const order = entry.order || [];
    const params = [wr !== null && wr !== undefined ? `wᵣ = ${trim(wr)}` : null,
      beam ? `beam width = ${beam}` : null].filter(Boolean).join(" · ");
    return `<article class="schedule-card ${method === "sweep_beam_D" ? "schedule-card--beam" : ""}">
      <div class="schedule-card__head">
        <div><span class="eyebrow">deterministic scheduler</span><h3>${escapeHtml(entry.label || method)}</h3></div>
        <span class="schedule-count">${int(order.length)} queries</span>
      </div>
      <div class="primary-metrics">
        ${metric("Modeled hits", int(entry.total_hits))}
        ${metric("Modeled misses", int(entry.total_misses))}
        ${metric("F<sub>hit</sub>", pct(entry.f_hit, 2), "metric-value--accent")}
      </div>
      <div class="schedule-params">${params || "Prepared scheduler artifact"}</div>
      <div class="order-label"><span>Complete selected order</span><span>${int(entry.total_requests)} page requests</span></div>
      <div class="order-ribbon" aria-label="complete selected query order">${order.map((q, i) => `<span class="query-pill">${escapeHtml(q)}</span>${i < order.length - 1 ? `<span class="order-arrow">→</span>` : ""}`).join("")}</div>
      <div class="schedule-card__actions">
        <button class="btn btn--primary" data-explore="${method}">Explore this schedule</button>
        <button class="btn" data-explain="${method}">Explain this schedule</button>
      </div>
    </article>`;
  }
}

function metric(label, value, cls = "") {
  return `<div class="primary-metric"><span>${label}</span><strong class="num ${cls}">${value}</strong></div>`;
}
function field(label, control) {
  const wrap = document.createElement("label"); wrap.className = "field";
  const span = document.createElement("span"); span.className = "eyebrow"; span.textContent = label;
  wrap.append(span, control); return wrap;
}
function option(value, label) { const o = document.createElement("option"); o.value = value; o.textContent = label; return o; }
function trim(x) { return Number(x).toFixed(2).replace(/0+$/, "").replace(/\.$/, ""); }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c])); }
