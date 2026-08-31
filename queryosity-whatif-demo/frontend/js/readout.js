// Research-facing live scoring readout. The three primary quantities are the
// same as the paper: modeled hits, modeled misses, and F_hit. Requests and
// capacity are supporting context; simulator wall-clock latency is deliberately
// not presented as a research metric.

import { int, pct, pp, signed } from "./format.js";

export function renderReadout(mount, opts) {
  const {
    current = null,
    baseline = null,
    capacity = null,
    simulating = false,
    error = null,
  } = opts;

  if (error) {
    mount.innerHTML = `<div class="score-readout panel">
      <div class="spread"><span class="eyebrow">live cache rescoring</span><span class="status-chip status-chip--error">error</span></div>
      <div class="err-inline">${escapeHtml(error.message)}${error.code ? ` <code>${escapeHtml(error.code)}</code>` : ""}</div>
    </div>`;
    return;
  }

  const f = current ? ratio(current) : null;
  mount.innerHTML = `<div class="score-readout panel ${simulating ? "is-simulating" : ""}">
    <div class="spread score-readout__head">
      <div><span class="eyebrow">live cache rescore</span><h3>Modeled effect of edited schedule</h3></div>
      <span class="status-chip">${simulating ? "rescoring…" : "simulator only"}</span>
    </div>
    <div class="live-primary-metrics">
      ${metric("Modeled hits", current ? int(current.total_hits) : "—", "hit")}
      ${metric("Modeled misses", current ? int(current.total_misses) : "—")}
      ${metric("F<sub>hit</sub>", f === null ? "—" : pct(f, 2), "metric-value--accent")}
    </div>
    <div class="score-supporting">
      <span><b>${current ? int(current.total_requests) : "—"}</b> page requests</span>
      <span><b>${capacity ? int(capacity.pages) : "—"}</b> buffer pages${capacity ? ` · ${escapeHtml(capacity.human)}` : ""}</span>
      <span>one complete-order simulation</span>
    </div>
    ${current && baseline && baseline.score ? deltaCard(baseline, current) : ""}
    <div class="inline-semantics"><strong>Editing does not rerun Sweep-D or Sweep+Beam-D.</strong> The order is sent directly to the shared deterministic cache simulator.</div>
  </div>`;
}

function metric(label, value, cls = "") {
  return `<div class="live-metric"><span>${label}</span><strong class="num ${cls}">${value}</strong></div>`;
}

function deltaCard(baseline, cur) {
  const base = baseline.score;
  const baseF = ratio(base);
  const curF = ratio(cur);
  const dH = cur.total_hits - base.total_hits;
  const dM = cur.total_misses - base.total_misses;
  const dF = curF - baseF;
  const direction = dF > 0 ? "up" : dF < 0 ? "down" : "flat";
  const comparedWith = baseline.isGenerated
    ? `Compared with loaded ${baseline.label} schedule`
    : `Compared with ${baseline.label}`;

  return `<div class="generated-comparison">
    <div class="comparison-title"><span class="eyebrow">comparison</span><strong>${escapeHtml(comparedWith)}</strong></div>
    <div class="comparison-grid">
      <div class="comparison-stat">
        <span>Generated / baseline schedule</span>
        <small>${escapeHtml(baseline.label)}</small>
        <strong class="num">${pct(baseF, 2)}</strong>
        <em>F<sub>hit</sub></em>
      </div>
      <div class="comparison-stat comparison-stat--edited">
        <span>Edited schedule</span>
        <small>current order</small>
        <strong class="num">${pct(curF, 2)}</strong>
        <em>F<sub>hit</sub></em>
      </div>
      <div class="comparison-stat comparison-stat--delta ${direction}">
        <span>Difference</span>
        <small>edited − baseline</small>
        <strong class="num">${pp(dF, 2)}</strong>
        <em>ΔF<sub>hit</sub></em>
      </div>
    </div>
    <div class="delta-badges">
      <span class="dbadge ${dH > 0 ? "up" : dH < 0 ? "down" : "flat"}">Δ hits ${signed(dH)}</span>
      <span class="dbadge ${dM < 0 ? "up" : dM > 0 ? "down" : "flat"}">Δ misses ${signed(dM)}</span>
    </div>
    <p class="delta-line">${deltaSentence(baseF, curF, dH, cur.total_requests)}</p>
  </div>`;
}

function ratio(x) {
  if (x.f_hit !== undefined && x.f_hit !== null) return x.f_hit;
  return x.hit_ratio;
}

function deltaSentence(baseF, curF, dH, requests) {
  if (Math.abs(curF - baseF) < 1e-12) {
    return `The edited order has the same modeled F_hit as the comparison schedule (${pct(curF, 2)}).`;
  }
  const verb = curF > baseF ? "higher" : "lower";
  const hits = dH === 0 ? "the same number of modeled hits" : `${int(Math.abs(dH))} ${dH > 0 ? "more" : "fewer"} modeled hits`;
  return `The edited order has ${verb} modeled F_hit (${pct(baseF, 2)} → ${pct(curF, 2)}), with ${hits} across ${int(requests)} page requests. This comparison does not imply global optimality.`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
