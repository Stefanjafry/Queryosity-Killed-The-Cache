// Queryosity conference-demo bootstrap. The UI has the same three primary
// objectives as the demo paper: capacity-aware planning, interactive schedule
// exploration, and explanation of generated schedules.

import { Api, Scorer } from "./api.js";
import { SchedulerView } from "./scheduler_view.js";
import { WhatIfWorkspace } from "./whatif.js";
import { ExplainView } from "./explain_view.js";
import { renderMethodology } from "./methodology.js";

const el = (id) => document.getElementById(id);

const dom = {
  loading: el("app-loading"),
  topbar: document.querySelector(".topbar"),
  views: el("views"),
  nav: el("nav"),
  badge: el("backend-badge"),
  themeBtn: el("theme-toggle"),
  methBtn: el("open-methodology"),
  drawer: el("methodology-drawer"),
  methBody: el("methodology-body"),
  fatal: el("fatal"),
  planning: el("view-planning"),
  exploration: el("view-exploration"),
  explain: el("view-explain"),
};

boot();

async function boot() {
  let meta;
  let health;
  try {
    [meta, health] = await Promise.all([
      Api.workloads(),
      Api.health().catch(() => null),
    ]);
  } catch (err) {
    return fatal(
      "Could not reach the Queryosity backend.",
      `${err.code || "error"}: ${err.message}. Is the Flask server running on 127.0.0.1:8000?`,
    );
  }

  if (!meta.workloads || !meta.workloads.length) {
    return fatal("No workloads are loaded.", "The backend started but reported no workload profiles.");
  }

  const backend = meta.backend;
  const fallbackReason = health && health.fallback_reason;
  setBadge(backend);
  setupThemeToggle();
  setupMethodology(backend, meta.wired, fallbackReason, meta.page_size_bytes);

  const scorer = new Scorer();
  const presets = meta.presets.map((p) => ({ pages: p.pages, human: p.human }));
  const planningWorkloads = meta.workloads.map((w) => ({ name: w.name, label: w.label }));

  const workspace = new WhatIfWorkspace({
    mount: dom.exploration,
    workloads: meta.workloads,
    presets,
    scorer,
  });

  const explainView = new ExplainView({
    mount: dom.explain,
    workloads: planningWorkloads,
    presets,
  });

  // eslint-disable-next-line no-new
  new SchedulerView({
    mount: dom.planning,
    workloads: planningWorkloads,
    presets,
    onExplore: (workload, cap, order, label, method) => {
      workspace.loadExternalOrder(workload, cap, order, label, method);
      showView("exploration");
    },
    onExplain: (workload, cap, method) => {
      explainView.setContext(workload, cap, method);
      showView("explain");
    },
  });

  wireNav();
  reveal();
}

function setBadge(backend) {
  dom.badge.classList.remove("badge--unknown");
  if (backend === "real") {
    dom.badge.classList.add("badge--real");
    dom.badge.textContent = "research profiles";
    dom.badge.title = "Live rescoring uses the Queryosity page-level cache simulator and stored research profiles.";
  } else {
    dom.badge.classList.add("badge--mock");
    dom.badge.textContent = "mock data";
    dom.badge.title = "Synthetic development data. Not paper-valid benchmark results.";
  }
}

function setupThemeToggle() {
  const key = "queryosity-paper-light";
  let light = false;
  try { light = window.localStorage.getItem(key) === "1"; } catch (_) { /* storage may be unavailable */ }
  const apply = () => {
    document.body.classList.toggle("theme-light", light);
    dom.themeBtn.textContent = light ? "Dark theme" : "Paper light";
    dom.themeBtn.setAttribute("aria-pressed", light ? "true" : "false");
  };
  apply();
  dom.themeBtn.addEventListener("click", () => {
    light = !light;
    try { window.localStorage.setItem(key, light ? "1" : "0"); } catch (_) { /* no-op */ }
    apply();
  });
}

function setupMethodology(backend, wired, fallbackReason, pageSizeBytes) {
  renderMethodology(dom.methBody, {
    backend,
    wired,
    fallbackReason,
    pageSizeNote: `${pageSizeBytes.toLocaleString("en-US")} bytes per page`,
  });

  const open = () => {
    dom.drawer.hidden = false;
    dom.drawer.setAttribute("aria-hidden", "false");
  };
  const close = () => {
    dom.drawer.hidden = true;
    dom.drawer.setAttribute("aria-hidden", "true");
  };
  dom.methBtn.addEventListener("click", open);
  dom.drawer.querySelectorAll("[data-close-methodology]").forEach((n) => n.addEventListener("click", close));
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !dom.drawer.hidden) close();
  });
}

function wireNav() {
  dom.nav.querySelectorAll(".nav-item").forEach((btn) =>
    btn.addEventListener("click", () => showView(btn.getAttribute("data-view"))));
  showView("planning");
}

function showView(view) {
  for (const v of ["planning", "exploration", "explain"]) {
    const section = el(`view-${v}`);
    if (section) section.hidden = v !== view;
  }
  dom.nav.querySelectorAll(".nav-item").forEach((btn) =>
    btn.classList.toggle("is-active", btn.getAttribute("data-view") === view));
  window.scrollTo({ top: 0, behavior: "auto" });
}

function reveal() {
  dom.loading.hidden = true;
  dom.loading.style.display = "none";
  dom.topbar.hidden = false;
  dom.views.hidden = false;
}

function fatal(title, detail) {
  dom.loading.style.display = "none";
  dom.fatal.hidden = false;
  dom.fatal.innerHTML = `<h2>${escapeHtml(title)}</h2><p>${escapeHtml(detail)}</p>`;
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
