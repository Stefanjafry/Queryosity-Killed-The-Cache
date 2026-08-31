// Focused subset-selection modal. Lets the user pick a small set of queries
// directly instead of loading the full workload and deleting the rest. Calls
// onCreate(selectedIdsInWorkloadOrder) when confirmed.

import { int } from "./format.js";

export function openSubsetBuilder({ allIds, pageCounts = {}, preselected = [], onCreate }) {
  const selected = new Set(preselected.filter((id) => allIds.includes(id)));
  let filter = "";

  const root = document.createElement("div");
  root.className = "modal";
  root.innerHTML = `
    <div class="modal__scrim" data-close></div>
    <div class="modal__panel" role="dialog" aria-modal="true" aria-label="Edit queries">
      <button class="drawer__close" data-close aria-label="Close">\u2715</button>
      <h2 class="modal__title">Edit queries</h2>
      <p class="hint">Check the queries to include. Your current order is kept for the ones you keep; newly added queries go to the end.</p>
      <div class="modal__controls">
        <input type="text" class="subset-search" placeholder="Search queries\u2026" aria-label="Search queries" />
        <span class="subset-count num"></span>
      </div>
      <div class="btngroup modal__actions">
        <button class="btn btn--sm" data-act="clear">Clear</button>
        <button class="btn btn--sm" data-act="all">Select all</button>
        <button class="btn btn--sm" data-act="visible">Select visible</button>
        <button class="btn btn--sm" data-act="first5">First 5</button>
        <button class="btn btn--sm" data-act="rand5">Random 5</button>
        <span class="subset-randn">
          <button class="btn btn--sm" data-act="randn">Random</button>
          <input type="number" class="num subset-n" min="1" value="10" aria-label="random N count" />
        </span>
      </div>
      <ul class="subset-list" role="group" aria-label="queries"></ul>
      <div class="modal__foot">
        <button class="btn" data-close>Cancel</button>
        <button class="btn btn--primary" data-create>Create schedule</button>
      </div>
    </div>`;
  document.body.appendChild(root);

  const listEl = root.querySelector(".subset-list");
  const countEl = root.querySelector(".subset-count");
  const searchEl = root.querySelector(".subset-search");
  const nEl = root.querySelector(".subset-n");
  const createBtn = root.querySelector("[data-create]");

  const matches = (id) => !filter || String(id).toLowerCase().includes(filter);
  const visibleIds = () => allIds.filter(matches);

  function renderList() {
    const frag = document.createDocumentFragment();
    for (const id of visibleIds()) {
      const li = document.createElement("li");
      const lab = document.createElement("label");
      lab.className = "subset-item" + (selected.has(id) ? " is-checked" : "");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = selected.has(id);
      cb.addEventListener("change", () => {
        if (cb.checked) selected.add(id); else selected.delete(id);
        lab.classList.toggle("is-checked", cb.checked);
        updateCount();
      });
      const name = document.createElement("span");
      name.className = "subset-item__id";
      name.textContent = id;
      const pg = document.createElement("span");
      pg.className = "subset-item__pg num";
      pg.textContent = pageCounts[id] === undefined ? "" : `${int(pageCounts[id])} pg`;
      lab.append(cb, name, pg);
      li.append(lab);
      frag.append(li);
    }
    listEl.replaceChildren(frag);
    if (!visibleIds().length) {
      const e = document.createElement("div");
      e.className = "library__empty";
      e.textContent = "No queries match that search.";
      listEl.replaceChildren(e);
    }
  }

  function updateCount() {
    countEl.textContent = `${selected.size} of ${allIds.length} selected`;
    createBtn.disabled = selected.size === 0;
    createBtn.textContent = `Update schedule \u00b7 ${selected.size} ${selected.size === 1 ? "query" : "queries"}`;
  }

  function sample(n) {
    const pool = allIds.slice();
    for (let i = pool.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    selected.clear();
    pool.slice(0, Math.max(0, Math.min(n, pool.length))).forEach((id) => selected.add(id));
  }

  searchEl.addEventListener("input", () => { filter = searchEl.value.trim().toLowerCase(); renderList(); });

  root.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", () => {
    const act = b.getAttribute("data-act");
    if (act === "clear") selected.clear();
    else if (act === "all") allIds.forEach((id) => selected.add(id));
    else if (act === "visible") visibleIds().forEach((id) => selected.add(id));
    else if (act === "first5") { selected.clear(); allIds.slice(0, 5).forEach((id) => selected.add(id)); }
    else if (act === "rand5") sample(5);
    else if (act === "randn") sample(parseInt(nEl.value, 10) || 0);
    renderList(); updateCount();
  }));

  function close() { root.remove(); document.removeEventListener("keydown", onKey); }
  function onKey(e) { if (e.key === "Escape") close(); }
  root.querySelectorAll("[data-close]").forEach((n) => n.addEventListener("click", close));
  document.addEventListener("keydown", onKey);

  createBtn.addEventListener("click", () => {
    if (!selected.size) return;
    const ordered = allIds.filter((id) => selected.has(id)); // workload order
    close();
    onCreate(ordered);
  });

  renderList();
  updateCount();
  searchEl.focus();
}
