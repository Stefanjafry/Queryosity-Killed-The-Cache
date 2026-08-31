// Methodology drawer: paper-consistent semantics for profiling, generated
// schedules, construction scores, complete-order F_hit, and live rescoring.

export function renderMethodology(container, info) {
  const backendLine = info.backend === "real"
    ? `<span class="num" style="color:var(--cyan)">research</span> — stored workload profiles scored by the Queryosity simulator.`
    : `<span class="num" style="color:var(--amber)">mock</span> — synthetic development profiles and illustrative generated schedules.`;

  const fallback = info.fallbackReason
    ? `<div class="callout warn">The research simulator could not be loaded, so this process fell back to mock mode: <code>${escapeHtml(info.fallbackReason)}</code></div>`
    : "";

  container.innerHTML = `<div class="doc">
    <h2>How Queryosity works</h2>
    <p>Active data source: ${backendLine}</p>
    ${fallback}

    <h3>1. Stored residency profiles</h3>
    <p>PostgreSQL is used before the conference interaction to profile each recurring read-only query independently. Queryosity records the pages left resident after the query completes. The live demo operates from those stored profiles; it does not execute schedules in PostgreSQL.</p>

    <h3>2. Capacity-aware schedule generation</h3>
    <p>At buffer capacity <em>C</em>, Queryosity derives directional reuse <em>D(C)</em> with its page-level clock-sweep simulator. Sweep-D and Sweep+Beam-D use normalized directional reuse and regret to construct candidate orders. Complete candidates are then replayed through the same simulator, and each method retains its highest-<em>F</em><sub>hit</sub> candidate.</p>
    <div class="callout"><strong>Construction score ≠ complete-order score.</strong> Directional reuse, regret, and transition score guide candidate construction. <em>F</em><sub>hit</sub> from complete-order cache simulation selects among completed candidates.</div>

    <h3>3. Interactive editing</h3>
    <p>Dragging, adding, removing, reversing, or importing queries changes only the order supplied to the live scorer. The edited order goes directly to the shared cache simulator and returns modeled hits, modeled misses, and <em>F</em><sub>hit</sub>. It does <strong>not</strong> rerun Sweep-D or Sweep+Beam-D.</p>

    <h3>What F<sub>hit</sub> means</h3>
    <p><em>F</em><sub>hit</sub>(π,C) = modeled page hits / modeled page requests for schedule π at capacity C. It is a cache-reuse score, not a runtime predictor. Queryosity does not claim that the simulator exactly reproduces PostgreSQL, and runtime is not substituted for the simulator score in this interface.</p>

    <h3>Prepared capacities</h3>
    <p>The conference configurations use 102,400 / 262,144 / 524,288 pages (${escapeHtml(info.pageSizeNote)}), approximately 800 MB / 2 GB / 4 GB. Generated schedules and explanations are capacity-specific.</p>

    <h3>Small-subset reference</h3>
    <p>Schedule Exploration includes an optional exhaustive reference for a deliberately small selected subset. It enumerates every permutation of that subset. It is separate from Sweep-D and Sweep+Beam-D and is not a global optimum for the full benchmark workload.</p>

    ${info.backend === "mock" ? `<h3>Mock-mode boundary</h3><div class="callout warn">Mock schedules are clearly marked and are not paper-valid results. Transition-level scheduler explanations are intentionally unavailable in mock mode rather than being fabricated. Generate the real artifacts on the Queryosity VM to enable Objective 3.</div>` : ""}
  </div>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
