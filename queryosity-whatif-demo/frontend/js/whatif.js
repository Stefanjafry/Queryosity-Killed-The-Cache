// Objective 2: interactive schedule exploration. Build or edit an order and
// send it directly to the shared simulator; scheduler search is never invoked. Owns
// the ScheduleState + wrapped board + readout and the live-scoring lifecycle
// (debounced rescoring, a "simulating" state, guarded controls, a manual Score
// button, and token-based stale-response protection).

import { Api, debounce } from "./api.js";
import { ScheduleState } from "./state.js";
import { ScheduleBoard } from "./schedule_board.js";
import { openSubsetBuilder } from "./subset_builder.js";
import { renderReadout } from "./readout.js";
import { int, bytes } from "./format.js";

export class WhatIfWorkspace {
  constructor({ mount, workloads, presets, scorer }) {
    this.mount = mount;
    this.workloads = workloads;
    this.byName = Object.fromEntries(workloads.map((w) => [w.name, w]));
    this.presets = presets;
    this.scorer = scorer;

    this.workload = workloads[0].name;
    this.capacityPages = presets[0].pages;

    this.current = null;
    this.baselineScore = null;
    this.currentFromCache = false;
    this.simulating = false;
    this.error = null;
    this._active = null;
    this.loadedGenerated = null;
    this._exactResult = null;
    this._exactBusy = false;
    this._exactError = null;
    this._exactOpen = false;

    this._scoreToken = 0;
    this._stateUnsub = null;
    this._debouncedRefresh = debounce(() => this.refreshScores(), 250);

    this._build();
    this._setWorkload(this.workload, { initial: true });
  }

  // -- skeleton ------------------------------------------------------------

  _build() {
    this.mount.innerHTML = `
      <div class="section-title section-title--compact">
        <span class="objective-kicker">Objective 2</span>
        <h2>Interactive Schedule Exploration</h2>
        <p>Load a generated Queryosity schedule or build your own order, then drag, add,
        remove, randomize, import, or export queries. Each edit is rescored directly by
        the shared cache simulator as modeled hits, modeled misses, and F<sub>hit</sub>.
        Sweep-D and Sweep+Beam-D are not rerun.</p>
      </div>

      <div class="wf-config" id="wf-config"></div>
      <div class="notice" id="wf-capnote" hidden></div>
      <div class="wf-status" id="wf-status"></div>
      <div class="wf-toolbar toolbar" id="wf-toolbar"></div>
      <div id="wf-picker"></div>
      <div class="wf-bulk" id="wf-bulk" hidden></div>

      <div id="wf-source" class="loaded-source"></div>

      <div class="wf-grid">
        <section class="wf-board-wrap panel" aria-label="Editable schedule">
          <div class="wf-board-head">
            <div><span class="eyebrow">edit schedule</span><h3>Editable schedule</h3></div>
            <span class="mono-note">drag tiles to reorder</span>
          </div>
          <div class="board" id="wf-board" role="list" aria-label="query schedule"></div>
          <div class="board-empty" id="wf-empty" hidden></div>
        </section>
        <aside class="wf-side" aria-label="Modeled effect">
          <div class="readout" id="wf-readout"></div>
          <div id="wf-inspector" class="inspector"></div>
        </aside>
      </div>

      <div id="wf-exact" class="exact-reference"></div>
      <div id="wf-live" class="sr-live" aria-live="polite"></div>`;

    this.configEl = this.mount.querySelector("#wf-config");
    this.capNoteEl = this.mount.querySelector("#wf-capnote");
    this.statusEl = this.mount.querySelector("#wf-status");
    this.toolbarEl = this.mount.querySelector("#wf-toolbar");
    this.pickerEl = this.mount.querySelector("#wf-picker");
    this.bulkEl = this.mount.querySelector("#wf-bulk");
    this.boardEl = this.mount.querySelector("#wf-board");
    this.emptyEl = this.mount.querySelector("#wf-empty");
    this.sourceEl = this.mount.querySelector("#wf-source");
    this.inspectorEl = this.mount.querySelector("#wf-inspector");
    this.readoutEl = this.mount.querySelector("#wf-readout");
    this.exactEl = this.mount.querySelector("#wf-exact");
    this.liveEl = this.mount.querySelector("#wf-live");

    this._buildConfig();
    this._buildToolbar();
  }

  _buildConfig() {
    const wsel = document.createElement("select");
    for (const w of this.workloads) {
      const o = document.createElement("option");
      o.value = w.name; o.textContent = w.label;
      wsel.append(o);
    }
    wsel.value = this.workload;
    wsel.addEventListener("change", () => this._setWorkload(wsel.value));
    this._workloadSelect = wsel;

    const csel = document.createElement("select");
    for (const p of this.presets) {
      const o = document.createElement("option");
      o.value = String(p.pages);
      o.textContent = `${int(p.pages)} pages \u00b7 ${p.human}`;
      csel.append(o);
    }
    csel.value = String(this.capacityPages);
    csel.addEventListener("change", () => {
      this.capacityPages = parseInt(csel.value, 10);
      this._afterCapacityChange();
    });
    this._capacitySelect = csel;

    this.configEl.append(
      field("Workload", wsel),
      field("Buffer capacity", csel)
    );
  }

  _buildToolbar() {
    const mk = (label, title, onClick, cls = "btn") => {
      const b = document.createElement("button");
      b.className = cls; b.textContent = label; b.title = title || label;
      b.addEventListener("click", onClick);
      return b;
    };

    // Primary demo flow: load → edit → rescore → restore/compare.
    this._btnLoadGenerated = mk("Load generated schedule…", "Load Sweep-D or Sweep+Beam-D for this workload and capacity",
      () => this._toggleLoadPicker(), "btn btn--primary");
    this._btnScore = mk("Score / Rescore", "Send the current order directly to the cache simulator",
      () => { this._debouncedRefresh.cancel(); this.refreshScores(); }, "btn btn--primary");
    this._btnRestore = mk("Restore original", "Restore the loaded generated schedule, or the natural order when no generated schedule is loaded",
      () => this._restoreLoadedOrOriginal(), "btn btn--emphasis");

    // Secondary editing/comparison controls remain available but visually quieter.
    this._btnUndo = mk("Undo", "Undo the last change", () => this.state.undo(), "btn btn--secondary");
    this._btnRedo = mk("Redo", "Redo", () => this.state.redo(), "btn btn--secondary");
    this._btnReverse = mk("Reverse", "Reverse the current order", () => this.state.reverse(), "btn btn--secondary");
    this._btnRandom = mk("Randomize", "Shuffle the current order", () => this.state.randomize(), "btn btn--secondary");
    this._btnSaveBase = mk("Save comparison", "Make the current order the comparison schedule",
      () => { this.state.saveBaseline(`Saved comparison (${this.state.order.length} queries)`); this._announce("Comparison schedule saved"); }, "btn btn--secondary");

    const moreWrap = document.createElement("div");
    moreWrap.className = "menu-wrap";
    const moreBtn = mk("More ▾", "More actions", () => this._toggleMore(), "btn btn--ghost btn--secondary");
    const menu = document.createElement("div");
    menu.className = "menu"; menu.hidden = true;
    const mitem = (label, onClick) => {
      const b = document.createElement("button");
      b.className = "menu__item"; b.textContent = label;
      b.addEventListener("click", () => { menu.hidden = true; onClick(); });
      return b;
    };
    this._btnResetBase = mitem("Reset comparison", () => { this.state.resetBaselineToOriginal(); this._announce("Comparison reset to natural order"); });
    menu.append(
      mitem("Clear schedule", () => this.state.setOrder([], "clear")),
      mitem("Load generated schedule…", () => this._toggleLoadPicker()),
      mitem("Restore natural full order", () => this._setScopeFull()),
      mitem("Import…", () => this._importInput.click()),
      mitem("Export", () => this._export()),
      this._btnResetBase
    );
    moreWrap.append(moreBtn, menu);
    this._moreMenu = menu;
    document.addEventListener("click", (e) => {
      if (!moreWrap.contains(e.target)) menu.hidden = true;
    });

    const primary = document.createElement("div");
    primary.className = "toolbar-group toolbar-group--primary";
    primary.append(this._btnLoadGenerated, this._btnScore, this._btnRestore);
    const secondary = document.createElement("div");
    secondary.className = "toolbar-group toolbar-group--secondary";
    secondary.append(this._btnUndo, this._btnRedo, this._btnReverse, this._btnRandom, this._btnSaveBase, moreWrap);
    this.toolbarEl.append(primary, secondary);

    const imp = document.createElement("input");
    imp.type = "file"; imp.accept = "application/json,.json"; imp.hidden = true;
    imp.addEventListener("change", () => { const f = imp.files && imp.files[0]; if (f) this._import(f); imp.value = ""; });
    this._importInput = imp;
    this.toolbarEl.append(imp);
  }

  _restoreLoadedOrOriginal() {
    if (this.loadedGenerated) {
      const { order, label } = this.loadedGenerated;
      this.state.baseline = { order: order.slice(), label };
      this.state.loadOrder(order.slice(), "restore-generated");
      this._announce(`${label} restored`);
      return;
    }
    this.state.baseline = { order: this.state.original.slice(), label: "Original order" };
    this.state.restoreOriginal();
    this._announce("Original workload order restored");
  }

  _toggleMore() { this._moreMenu.hidden = !this._moreMenu.hidden; }

  // -- workload / capacity -------------------------------------------------

  _setWorkload(name) {
    if (!this.byName[name]) return;
    this.workload = name;
    if (this._workloadSelect) this._workloadSelect.value = name;

    const w = this.byName[name];
    const allIds = w.query_ids.slice();

    if (this.board) this.board.destroy();
    if (this._stateUnsub) this._stateUnsub();

    this.state = new ScheduleState(name, allIds);
    this._active = null;
    this.loadedGenerated = null;
    this._exactResult = null;
    this._exactError = null;
    this._stateUnsub = this.state.subscribe((reason) => this._onStateChange(reason));
    this.board = new ScheduleBoard({
      boardEl: this.boardEl,
      state: this.state,
      pageCounts: w.page_counts || {},
      liveEl: this.liveEl,
      onActiveChange: (id) => { this._active = id; this._renderInspector(); },
    });

    this._closeLoadPicker();
    this._hideCapNote();
    this._renderAll();
    this.refreshScores();
  }

  _afterCapacityChange() {
    if (this._capacitySelect) this._capacitySelect.value = String(this.capacityPages);
    this._closeLoadPicker();
    this._exactResult = null;
    this._exactError = null;
    this._renderExactReference();
    if (this.loadedGenerated) {
      this.loadedGenerated = null;
      this.state.resetBaselineToOriginal();
      this._renderSource();
      this._announce("Capacity changed. Load the generated schedule for this capacity to restore a paper-valid comparison.");
    }
    this._debouncedRefresh.cancel();
    this.refreshScores();
  }

  // -- scope ---------------------------------------------------------------

  _isFull() { return this.state.order.length === this.state.allIds.length; }

  _setScopeFull() {
    if (!this._isFull()) {
      const ok = window.confirm(
        `Load all ${this.state.allIds.length} queries in natural order? This replaces the current schedule.`);
      if (!ok) return;
    }
    this.state.restoreOriginal();
    this._announce("Full workload loaded");
  }

  _openSubset() {
    const w = this.byName[this.workload];
    openSubsetBuilder({
      allIds: this.state.allIds,
      pageCounts: w.page_counts || {},
      preselected: this.state.order,
      onCreate: (ids) => {
        // Order-preserving: keep currently-scheduled picks in place, append new
        // ones (in workload order), drop the unpicked.
        const sel = new Set(ids);
        const kept = this.state.order.filter((id) => sel.has(id));
        const added = ids.filter((id) => !this.state.order.includes(id));
        this.state.setOrder([...kept, ...added], "subset");
        this._announce(`Schedule set to ${ids.length} queries`);
      },
    });
  }

  // -- state change --------------------------------------------------------

  _onStateChange(reason) {
    if (reason !== "selection" && reason !== "baseline") {
      this._exactResult = null;
      this._exactError = null;
    }
    this._renderStatus();
    this._renderSource();
    this._renderBulk();
    this._renderInspector();
    this._renderExactReference();
    this._updateToolbar();
    if (reason === "selection") return;
    if (reason === "baseline") { this.refreshScores(); return; }
    this._debouncedRefresh();
  }

  _renderAll() {
    this._renderStatus();
    this._renderSource();
    this._renderBulk();
    this._renderInspector();
    this._renderReadout();
    this._renderExactReference();
    this._updateToolbar();
  }

  // -- status line ---------------------------------------------------------

  _renderStatus() {
    const total = this.state.allIds.length;
    const inN = this.state.order.length;
    const excluded = this.state.missingIds();
    const edit = `<button class="btn btn--edit" id="wf-edit">Edit queries\u2026</button>`;

    let left;
    if (excluded.length === 0) {
      left = `All <b>${int(total)}</b> workload queries are scheduled.`;
    } else {
      const preview = excluded.slice(0, 6).join(", ") + (excluded.length > 6 ? "\u2026" : "");
      left = `<b>${int(inN)}</b> of ${int(total)} scheduled &middot; ` +
        `<span class="excluded">${int(excluded.length)} excluded:</span> <span class="mono-note">${escapeHtml(preview)}</span>`;
    }
    this.statusEl.innerHTML = `<div class="wf-status__row"><span>${left}</span>${edit}</div>`;
    const eb = this.statusEl.querySelector("#wf-edit");
    if (eb) eb.addEventListener("click", () => this._openSubset());

    this.emptyEl.hidden = inN !== 0;
    if (inN === 0) {
      this.emptyEl.innerHTML =
        `<p>The schedule is empty. <button class="btn btn--sm" id="wf-empty-full">Load full workload</button>
         or <button class="btn btn--sm" id="wf-empty-sub">Edit queries\u2026</button>.</p>`;
      const f = this.emptyEl.querySelector("#wf-empty-full");
      const s = this.emptyEl.querySelector("#wf-empty-sub");
      if (f) f.addEventListener("click", () => this._setScopeFull());
      if (s) s.addEventListener("click", () => this._openSubset());
    }
  }

  // -- bulk toolbar --------------------------------------------------------

  _renderBulk() {
    const sel = [...this.state.selection];
    if (sel.length === 0) { this.bulkEl.hidden = true; this.bulkEl.innerHTML = ""; return; }
    this.bulkEl.hidden = false;
    this.bulkEl.innerHTML = `
      <span class="bulk__count num">${sel.length} selected</span>
      <span class="mono-note">drag any selected tile to move the group</span>
      <span class="bulk__spacer"></span>
      <button class="btn btn--sm" data-b="keep">Keep only selected</button>
      <button class="btn btn--sm btn--danger" data-b="remove">Remove selected</button>
      <button class="btn btn--sm btn--ghost" data-b="clear">Clear selection</button>`;
    this.bulkEl.querySelector('[data-b="keep"]').addEventListener("click", () => this._keepOnly(sel));
    this.bulkEl.querySelector('[data-b="remove"]').addEventListener("click", () => this._removeSelected(sel));
    this.bulkEl.querySelector('[data-b="clear"]').addEventListener("click", () => this.state.clearSelection());
  }

  _keepOnly(sel) {
    const dropCount = this.state.order.length - sel.length;
    if (dropCount > 10 && !window.confirm(`Keep only ${sel.length} queries and remove the other ${dropCount}?`)) return;
    this.state.keepOnly(sel);
    this.state.clearSelection();
    this._announce(`Kept ${sel.length} queries`);
  }

  _removeSelected(sel) {
    if (sel.length > 10 && !window.confirm(`Remove ${sel.length} queries from the schedule?`)) return;
    this.state.removeQueries(sel);
    this._announce(`Removed ${sel.length} queries`);
  }

  // -- inspector -----------------------------------------------------------

  _renderInspector() {
    const id = this._active && this.state.order.includes(this._active) ? this._active : null;
    this.inspectorEl.classList.toggle("is-collapsed", !id);
    if (!id) {
      this.inspectorEl.innerHTML =
        `<div class="inspector__empty"><span class="eyebrow">query inspector</span>
         <p class="hint">Select a query to move it precisely.</p></div>`;
      return;
    }
    const pos = this.state.order.indexOf(id) + 1;
    const len = this.state.order.length;
    const pc = (this.byName[this.workload].page_counts || {})[id];
    this.inspectorEl.innerHTML = `
      <div class="spread"><span class="eyebrow">query inspector</span></div>
      <div class="inspector__id num">${escapeHtml(id)}</div>
      <div class="inspector__meta mono-note">position ${pos} of ${len}${pc !== undefined ? ` &middot; ${int(pc)} pages` : ""}</div>
      <div class="inspector__move">
        <label class="field">
          <span class="eyebrow">move to position</span>
          <span class="row">
            <input type="number" class="num" id="insp-pos" min="1" max="${len}" value="${pos}" aria-label="move ${escapeHtml(id)} to position" />
            <button class="btn btn--sm" id="insp-go">Move</button>
          </span>
        </label>
      </div>
      <div class="btngroup">
        <button class="btn btn--sm" data-m="first">\u21e4 First</button>
        <button class="btn btn--sm" data-m="left">\u2190 Left</button>
        <button class="btn btn--sm" data-m="right">Right \u2192</button>
        <button class="btn btn--sm" data-m="last">Last \u21e5</button>
      </div>
      <button class="btn btn--sm btn--danger inspector__remove" data-m="remove">Remove from schedule</button>`;

    const go = () => {
      const v = parseInt(this.inspectorEl.querySelector("#insp-pos").value, 10);
      if (Number.isInteger(v)) { this.state.moveTo(id, v - 1); this.board.focusTile(id); }
    };
    this.inspectorEl.querySelector("#insp-go").addEventListener("click", go);
    this.inspectorEl.querySelector("#insp-pos").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); go(); } });
    this.inspectorEl.querySelectorAll("[data-m]").forEach((b) => b.addEventListener("click", () => {
      const m = b.getAttribute("data-m");
      if (m === "first") this.state.moveTo(id, 0);
      else if (m === "left") this.state.moveBy(id, -1);
      else if (m === "right") this.state.moveBy(id, +1);
      else if (m === "last") this.state.moveTo(id, this.state.order.length - 1);
      else if (m === "remove") { this.state.removeQueries([id]); return; }
      this.board.focusTile(id);
    }));
  }

  // -- scoring -------------------------------------------------------------

  async refreshScores() {
    const wl = this.workload;
    const cap = this.capacityPages;
    const order = this.state.order.slice();
    const baseOrder = this.state.baseline.order.slice();

    if (order.length === 0) {
      this.current = null; this.baselineScore = null;
      this.error = { message: "The schedule is empty. Add at least one query to score." };
      this.simulating = false;
      this._renderReadout(); this._updateToolbar();
      return;
    }

    const token = ++this._scoreToken;
    this.simulating = true;
    this._renderReadout(); this._updateToolbar();

    try {
      const curP = this.scorer.get(wl, cap, order);
      const baseP = baseOrder.length ? this.scorer.get(wl, cap, baseOrder) : Promise.resolve(null);
      const [cur, base] = await Promise.all([curP, baseP]);
      if (token !== this._scoreToken) return;
      this.current = cur.result;
      this.currentFromCache = cur.fromCache;
      this.baselineScore = base ? base.result : null;
      this.error = null;
      this.simulating = false;
    } catch (err) {
      if (token !== this._scoreToken) return;
      this.error = { message: err.message, code: err.code };
      this.simulating = false;
    }
    this._renderReadout();
    this._updateToolbar();
  }

  _renderReadout() {
    const human = bytes(this.capacityPages * 8192);
    renderReadout(this.readoutEl, {
      current: this.current,
      baseline: {
        label: this.state.baseline.label,
        score: this.baselineScore,
        isGenerated: !!this.loadedGenerated && sameOrder(this.state.baseline.order, this.loadedGenerated.order),
      },
      capacity: { pages: this.capacityPages, human },
      simulating: this.simulating,
      error: this.error,
      cached: this.currentFromCache,
    });
  }

  _updateToolbar() {
    if (this._btnUndo) this._btnUndo.disabled = !this.state.canUndo();
    if (this._btnRedo) this._btnRedo.disabled = !this.state.canRedo();
    if (this._btnScore) this._btnScore.disabled = this.simulating;
    if (this._btnRestore) {
      this._btnRestore.textContent = this.loadedGenerated ? `Restore ${this.loadedGenerated.label}` : "Restore original";
      this._btnRestore.title = this.loadedGenerated
        ? `Restore the unchanged loaded ${this.loadedGenerated.label} schedule`
        : "Restore the natural full-workload order";
    }
    if (this._btnResetBase) this._btnResetBase.disabled = this.state.baseline.label === "Original order";
  }

  // -- load generated schedule picker --------------------------------------

  _toggleLoadPicker() {
    if (this.pickerEl.dataset.open === "1") { this._closeLoadPicker(); return; }
    this._openLoadPicker();
  }
  _closeLoadPicker() {
    if (!this.pickerEl) return;
    this.pickerEl.innerHTML = "";
    this.pickerEl.dataset.open = "0";
  }
  async _openLoadPicker() {
    this.pickerEl.dataset.open = "1";
    this.pickerEl.innerHTML = `<div class="notice">Loading generated Queryosity schedules\u2026</div>`;
    let data;
    try {
      data = await Api.schedulersFor(this.workload, this.capacityPages);
    } catch (err) {
      if (err.status === 404) {
        this.pickerEl.innerHTML =
          `<div class="notice">No generated Queryosity schedules for this workload at
          ${int(this.capacityPages)} pages. Generate them on the VM with
          <code>python scripts/generate_demo_artifacts.py --all</code>.</div>`;
      } else {
        this.pickerEl.innerHTML =
          `<div class="err-inline">Could not load generated schedules: ${escapeHtml(err.message)}</div>`;
      }
      return;
    }
    const methods = data.methods || {};
    const keys = Object.keys(methods);
    if (!keys.length) { this._closeLoadPicker(); return; }

    const sel = document.createElement("select");
    for (const m of keys) {
      const o = document.createElement("option");
      o.value = m;
      const f = methods[m].f_hit ?? methods[m].hit_ratio;
      o.textContent = `${methods[m].label || m} \u2014 F_hit ${(f * 100).toFixed(2)}%`;
      sel.append(o);
    }
    const loadBtn = document.createElement("button");
    loadBtn.className = "btn btn--sm btn--primary"; loadBtn.textContent = "Load + compare";
    loadBtn.addEventListener("click", () => {
      const m = sel.value;
      const entry = methods[m];
      this._loadGeneratedEntry(entry.order, entry.label || m, m);
      this._closeLoadPicker();
    });
    const baseBtn = document.createElement("button");
    baseBtn.className = "btn btn--sm"; baseBtn.textContent = "Set as comparison";
    baseBtn.title = "Compare your current order against this generated schedule";
    baseBtn.addEventListener("click", () => {
      const m = sel.value;
      this.state.baseline = { order: methods[m].order.slice(), label: methods[m].label || m };
      this.state._emit("baseline");
    });
    const closeBtn = document.createElement("button");
    closeBtn.className = "btn btn--sm btn--ghost"; closeBtn.textContent = "Close";
    closeBtn.addEventListener("click", () => this._closeLoadPicker());

    const wrap = document.createElement("div");
    wrap.className = "panel panel--tight";
    const rowEl = document.createElement("div");
    rowEl.className = "row";
    const lbl = document.createElement("span");
    lbl.className = "eyebrow"; lbl.textContent = "generated schedule";
    rowEl.append(lbl, sel, loadBtn, baseBtn, closeBtn);
    wrap.append(rowEl);
    this.pickerEl.replaceChildren(wrap);
  }

  _setCapNote(msg, isError) {
    this.capNoteEl.hidden = false;
    this.capNoteEl.classList.toggle("err-inline", !!isError);
    this.capNoteEl.classList.toggle("notice", !isError);
    this.capNoteEl.textContent = msg;
  }
  _hideCapNote() { this.capNoteEl.hidden = true; }
  _announce(msg) { if (this.liveEl) this.liveEl.textContent = msg; }

  // -- generated source / exact-subset reference --------------------------

  _renderSource() {
    if (!this.sourceEl) return;
    if (!this.loadedGenerated) {
      this.sourceEl.innerHTML = `<div class="inline-semantics"><strong>Source:</strong> custom/current workload order. Use <em>Load generated…</em> to compare edits against Sweep-D or Sweep+Beam-D.</div>`;
      return;
    }
    this.sourceEl.innerHTML = `<div class="loaded-source__inner"><span class="eyebrow">loaded generated schedule</span><strong>${escapeHtml(this.loadedGenerated.label)}</strong><span>Edits are compared against the unchanged generated order; scoring goes directly to the cache simulator.</span></div>`;
  }

  _renderExactReference() {
    if (!this.exactEl || !this.state) return;
    const n = this.state.order.length;
    const maxN = 5;
    if (n < 2) { this.exactEl.innerHTML = ""; return; }

    let body = "";
    if (n > maxN) {
      body = `<p>Select at most ${maxN} queries to enable exhaustive permutation search. This optional reference is separate from Sweep-D/Sweep+Beam-D and is not the full-workload optimum.</p>`;
    } else {
      let result = "";
      if (this._exactBusy) result = `<span class="mono-note">Enumerating ${n}! permutations…</span>`;
      else if (this._exactError) result = `<span class="err-inline">${escapeHtml(this._exactError)}</span>`;
      else if (this._exactResult) {
        result = `<div class="exact-result"><span><b>Exact subset F<sub>hit</sub></b> ${(this._exactResult.f_hit * 100).toFixed(2)}%</span><span>${int(this._exactResult.permutations_evaluated)} permutations evaluated</span><button class="btn btn--sm" id="exact-load">Load exact subset order</button></div>`;
      }
      body = `<div class="spread exact-reference__top"><div><span class="eyebrow">small-subset reference</span><h3>Exact best order for this selected subset</h3></div><button class="btn btn--sm" id="exact-run" ${this._exactBusy ? "disabled" : ""}>Find exact best order</button></div>
      <p>Enumerates all permutations of these ${n} queries only. It is not Sweep-D, not Sweep+Beam-D, and not a global optimum for the full benchmark workload.</p>${result}`;
    }

    this.exactEl.innerHTML = `<div class="panel panel--tight exact-reference__shell">
      <button class="exact-accordion" id="exact-toggle" aria-expanded="${this._exactOpen ? "true" : "false"}" aria-controls="exact-content">
        <span class="exact-accordion__label"><span class="eyebrow">small-subset reference</span><strong>Exact subset comparison</strong></span>
        <span class="exact-accordion__arrow" aria-hidden="true">${this._exactOpen ? "▾" : "▸"}</span>
      </button>
      <div id="exact-content" class="exact-reference__content" ${this._exactOpen ? "" : "hidden"}>${body}</div>
    </div>`;

    const toggle = this.exactEl.querySelector("#exact-toggle");
    if (toggle) toggle.addEventListener("click", () => {
      this._exactOpen = !this._exactOpen;
      this._renderExactReference();
    });

    const run = this.exactEl.querySelector("#exact-run");
    if (run) run.addEventListener("click", () => this._runExactSubset());
    const load = this.exactEl.querySelector("#exact-load");
    if (load) load.addEventListener("click", () => this.state.loadOrder(this._exactResult.order.slice(), "load-exact-subset"));
  }

  async _runExactSubset() {
    const order = this.state.order.slice();
    this._exactBusy = true; this._exactError = null; this._exactResult = null;
    this._exactOpen = true;
    this._renderExactReference();
    try {
      this._exactResult = await Api.optimizeSubset(this.workload, this.capacityPages, order);
    } catch (err) {
      this._exactError = err.message;
    } finally {
      this._exactBusy = false;
      this._renderExactReference();
    }
  }

  // -- import / export -----------------------------------------------------

  _export() {
    const obj = this.state.exportObject(this.capacityPages);
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${this.workload}-schedule-${this.capacityPages}.json`;
    document.body.append(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async _import(file) {
    let data;
    try { data = JSON.parse(await file.text()); }
    catch (err) { this._setCapNote(`Could not parse that file as JSON: ${err.message}`, true); return; }
    const order = Array.isArray(data.order) ? data.order.map(String) : null;
    if (!order || !order.length) { this._setCapNote("That file has no 'order' array of query ids.", true); return; }
    if (data.workload && data.workload !== this.workload && this.byName[data.workload]) this._setWorkload(data.workload);
    else if (data.workload && !this.byName[data.workload]) this._setCapNote(`Unknown workload "${data.workload}" in file; importing into ${this.workload}.`, true);
    if (Number.isInteger(data.capacity_pages) && data.capacity_pages > 0) {
      const preset = this.presets.find((p) => p.pages === data.capacity_pages);
      if (preset) { this.capacityPages = data.capacity_pages; if (this._capacitySelect) this._capacitySelect.value = String(data.capacity_pages); }
    }
    const known = new Set(this.state.allIds);
    const unknown = order.filter((id) => !known.has(id));
    this.state.loadOrder(order, "import");
    if (unknown.length) this._setCapNote(`Imported. ${unknown.length} id(s) not in ${this.workload} were skipped.`, true);
    else this._hideCapNote();
  }

  // -- external entry point (scheduler view) ------------------------------

  loadExternalOrder(workload, cap, order, label = "Generated schedule", method = null) {
    if (this.byName[workload] && workload !== this.workload) this._setWorkload(workload);
    const preset = this.presets.find((p) => p.pages === cap);
    if (preset) {
      this.capacityPages = cap;
      if (this._capacitySelect) this._capacitySelect.value = String(cap);
    }
    this._loadGeneratedEntry(order, label, method);
    this._debouncedRefresh.cancel();
    this.refreshScores();
  }

  _loadGeneratedEntry(order, label, method) {
    const clean = order.slice();
    this.loadedGenerated = { order: clean.slice(), label, method };
    this.state.baseline = { order: clean.slice(), label };
    this.state.loadOrder(clean, "load-generated");
    this._announce(`${label} loaded. Edits will be compared against this generated schedule.`);
  }
}

function field(label, control) {
  const wrap = document.createElement("label");
  wrap.className = "field";
  const span = document.createElement("span");
  span.className = "eyebrow";
  span.innerHTML = label;
  wrap.append(span, control);
  return wrap;
}

function sameOrder(a, b) {
  return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => x === b[i]);
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
