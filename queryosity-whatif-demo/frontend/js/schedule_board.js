// Wrapped schedule board. Tiles flow left-to-right, row by row (never a snake),
// wrapping by width so the whole schedule is visible with no internal scrollbar.
// Position numbers are the authoritative sequence. Reordering works by drag
// (with an explicit insertion caret + destination position) and by keyboard
// (focus a tile, Ctrl+Arrow to move, or use the inspector's move controls).
//
// Selection and active-tile changes are surfaced via callbacks so the workspace
// can render the query inspector and the bulk-action toolbar. Selection updates
// only toggle classes; a full re-render happens only on structural edits.

import { int, compact } from "./format.js";

// Pure: final post-removal insertion index for a single-tile drag.
export function dropToIndex(from, targetIdx, insertBefore, len) {
  let raw = insertBefore ? targetIdx : targetIdx + 1;
  let to = from < raw ? raw - 1 : raw;
  return Math.max(0, Math.min(to, len - 1));
}

export class ScheduleBoard {
  constructor({ boardEl, state, pageCounts, liveEl, onSelectionChange, onActiveChange }) {
    this.boardEl = boardEl;
    this.state = state;
    this.pageCounts = pageCounts || {};
    this.liveEl = liveEl || null;
    this.onSelectionChange = onSelectionChange || (() => {});
    this.onActiveChange = onActiveChange || (() => {});

    this._active = null;      // id of the inspector's focused query
    this._anchor = null;      // shift-range anchor
    this._focusIndex = 0;     // roving tabindex position
    this._dragId = null;
    this._dragBatch = null;
    this._dropIndex = null;   // pending post-removal insertion index (single)
    this._dropRaw = null;     // raw insertion slot for the label

    this._dropLabel = document.createElement("div");
    this._dropLabel.className = "drop-label";
    this._dropLabel.hidden = true;
    document.body.appendChild(this._dropLabel);

    this._unsub = state.subscribe((reason) => this._onState(reason));
    this._onKeyRef = (e) => this._onBoardKey(e);
    this.render();
  }

  destroy() {
    if (this._unsub) this._unsub();
    this._setAutoScroll(0);
    if (this._dropLabel && this._dropLabel.parentNode) this._dropLabel.remove();
  }

  // -- state --------------------------------------------------------------

  _onState(reason) {
    if (reason === "selection") {
      this._applySelectionClasses();
      this.onSelectionChange([...this.state.selection]);
      return;
    }
    // structural change -> keep active if still present
    if (this._active && !this.state.order.includes(this._active)) this._active = null;
    this.render();
  }

  // -- render -------------------------------------------------------------

  render() {
    const s = this.state;
    const frag = document.createDocumentFragment();
    this._focusIndex = Math.min(this._focusIndex, Math.max(0, s.order.length - 1));

    s.order.forEach((id, idx) => frag.append(this._tile(id, idx)));
    this.boardEl.replaceChildren(frag);
    this._applySelectionClasses();
    this.onActiveChange(this._active);
  }

  _tile(id, idx) {
    const s = this.state;
    const el = document.createElement("div");
    el.className = "tile";
    el.dataset.id = id;
    el.dataset.idx = String(idx);
    el.setAttribute("role", "listitem");
    el.setAttribute("draggable", "true");
    el.tabIndex = idx === this._focusIndex ? 0 : -1;

    const pc = this.pageCounts[id];
    el.setAttribute("aria-label",
      `position ${idx + 1}, query ${id}${pc !== undefined ? `, ${int(pc)} pages` : ""}`);

    const pos = document.createElement("span");
    pos.className = "tile__pos num";
    pos.textContent = String(idx + 1);

    const body = document.createElement("span");
    body.className = "tile__body";
    const idEl = document.createElement("span");
    idEl.className = "tile__id";
    idEl.textContent = id;
    const pages = document.createElement("span");
    pages.className = "tile__pages num";
    pages.textContent = pc === undefined ? "" : `${compact(pc)} pg`;
    body.append(idEl, pages);

    const x = document.createElement("button");
    x.className = "tile__x";
    x.type = "button";
    x.textContent = "\u2715";
    x.title = `Remove ${id} from schedule`;
    x.setAttribute("aria-label", `Remove ${id} from schedule`);
    x.addEventListener("click", (e) => { e.stopPropagation(); s.removeQueries([id]); this._announce(`${id} removed`); });

    const arrow = document.createElement("span");
    arrow.className = "tile__arrow";
    arrow.setAttribute("aria-hidden", "true");
    arrow.textContent = "\u2192";

    el.append(pos, body, x, arrow);

    el.addEventListener("click", (e) => this._onClick(e, id, idx));
    el.addEventListener("focus", () => { this._focusIndex = idx; this._setActive(id); });
    el.addEventListener("dragstart", (e) => this._onDragStart(e, id, el));
    el.addEventListener("dragend", () => this._onDragEnd());
    el.addEventListener("dragover", (e) => this._onDragOver(e, el, idx));
    el.addEventListener("drop", (e) => this._onDrop(e));
    el.addEventListener("keydown", (e) => this._onTileKey(e, id, idx));
    return el;
  }

  _applySelectionClasses() {
    const sel = this.state.selection;
    for (const el of this.boardEl.querySelectorAll(".tile")) {
      el.classList.toggle("is-selected", sel.has(el.dataset.id));
      el.classList.toggle("is-active", el.dataset.id === this._active);
    }
  }

  // -- selection / active --------------------------------------------------

  _setActive(id) {
    if (this._active === id) return;
    this._active = id;
    this._applySelectionClasses();
    this.onActiveChange(id);
  }

  _onClick(e, id, idx) {
    const s = this.state;
    this._focusIndex = idx;
    if (e.shiftKey && this._anchor) {
      s.selectRangeTo(id, this._anchor);
    } else if (e.ctrlKey || e.metaKey) {
      s.toggleSelect(id);
      this._anchor = id;
    } else {
      if (s.selection.size === 1 && s.selection.has(id)) s.clearSelection();
      else { s.selectOnly(id); this._anchor = id; }
    }
    this._setActive(id);
  }

  focusTile(id) {
    const el = this.boardEl.querySelector(`.tile[data-id="${cssEscape(id)}"]`);
    if (el) { this._focusIndex = Number(el.dataset.idx); el.focus(); }
  }

  // -- drag ---------------------------------------------------------------

  _onDragStart(e, id, el) {
    if (e.target.closest(".tile__x")) { e.preventDefault(); return; }
    this._dragId = id;
    this._dragBatch = this.state.selection.has(id) && this.state.selection.size > 1
      ? [...this.state.selection] : [id];
    el.classList.add("is-ghost");
    this.boardEl.classList.add("is-dragging");
    if (e.dataTransfer) { e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", id); }
    document.addEventListener("keydown", this._onKeyRef); // Escape cancels
  }

  _onDragOver(e, el, idx) {
    if (this._dragId === null) return;
    e.preventDefault();
    if (e.dataTransfer) e.dataTransfer.dropEffect = "move";
    const rect = el.getBoundingClientRect();
    const insertBefore = e.clientX < rect.left + rect.width / 2;

    for (const t of this.boardEl.querySelectorAll(".is-drop-before,.is-drop-after"))
      t.classList.remove("is-drop-before", "is-drop-after");
    el.classList.add(insertBefore ? "is-drop-before" : "is-drop-after");

    const from = this.state.order.indexOf(this._dragId);
    this._dropIndex = dropToIndex(from, idx, insertBefore, this.state.order.length);
    this._dropRaw = insertBefore ? idx : idx + 1;

    this._dropLabel.hidden = false;
    this._dropLabel.textContent = `drop at position ${this._dropIndex + 1}`;
    this._dropLabel.style.left = `${e.clientX + 14}px`;
    this._dropLabel.style.top = `${e.clientY + 14}px`;

    this._edgeAutoScroll(e.clientY);
  }

  _onDrop(e) {
    if (this._dragId === null) return;
    e.preventDefault();
    const batch = this._dragBatch || [this._dragId];
    if (batch.length > 1) {
      // For a group, insert before the raw slot within the remaining order.
      const set = new Set(batch);
      const rest = this.state.order.filter((id) => !set.has(id));
      const removedBefore = this.state.order.slice(0, this._dropRaw).filter((id) => set.has(id)).length;
      this.state.moveBatch(batch, Math.max(0, this._dropRaw - removedBefore));
      this._announce(`${batch.length} queries moved`);
    } else {
      this.state.moveTo(this._dragId, this._dropIndex);
      this._announce(`${this._dragId} moved to position ${this._dropIndex + 1}`);
    }
    this._endDrag();
  }

  _onDragEnd() { this._endDrag(); }

  _endDrag() {
    this._setAutoScroll(0);
    this._dropLabel.hidden = true;
    this.boardEl.classList.remove("is-dragging");
    for (const t of this.boardEl.querySelectorAll(".is-ghost,.is-drop-before,.is-drop-after"))
      t.classList.remove("is-ghost", "is-drop-before", "is-drop-after");
    this._dragId = null;
    this._dragBatch = null;
    this._dropIndex = null;
    this._dropRaw = null;
    document.removeEventListener("keydown", this._onKeyRef);
  }

  _onBoardKey(e) {
    if (e.key === "Escape" && this._dragId !== null) {
      // cancel the in-flight drag without reordering
      this._dragId = null; this._dragBatch = null;
      this._endDrag();
    }
  }

  // edge auto-scroll (page-level, since the board has no internal scroll)
  _edgeAutoScroll(clientY) {
    const EDGE = 70;
    let dir = 0;
    if (clientY < EDGE) dir = -1;
    else if (clientY > window.innerHeight - EDGE) dir = 1;
    this._setAutoScroll(dir);
  }
  _setAutoScroll(dir) {
    this._scrollDir = dir;
    if (!dir) { if (this._scrollRAF) { cancelAnimationFrame(this._scrollRAF); this._scrollRAF = null; } return; }
    if (this._scrollRAF) return;
    const tick = () => {
      if (!this._scrollDir) { this._scrollRAF = null; return; }
      window.scrollBy(0, this._scrollDir * 16);
      this._scrollRAF = requestAnimationFrame(tick);
    };
    this._scrollRAF = requestAnimationFrame(tick);
  }

  // -- keyboard on a tile --------------------------------------------------

  _columns() {
    // Infer columns from the first row's tiles sharing the top offset.
    const tiles = this.boardEl.querySelectorAll(".tile");
    if (!tiles.length) return 1;
    const top = tiles[0].offsetTop;
    let c = 0;
    for (const t of tiles) { if (t.offsetTop === top) c++; else break; }
    return Math.max(1, c);
  }

  _onTileKey(e, id, idx) {
    const s = this.state;
    const mod = e.ctrlKey || e.metaKey;
    const cols = this._columns();
    const focusAt = (i) => {
      const t = this.boardEl.querySelector(`.tile[data-idx="${i}"]`);
      if (t) { this._focusIndex = i; t.focus(); }
    };

    if (mod && (e.key === "ArrowLeft" || e.key === "ArrowRight")) {
      e.preventDefault();
      const delta = e.key === "ArrowLeft" ? -1 : 1;
      s.moveBy(id, delta);
      this._announce(`${id} moved to position ${Math.min(Math.max(idx + delta, 0), s.order.length - 1) + 1}`);
      requestAnimationFrame(() => this.focusTile(id));
      return;
    }
    if (mod && e.key === "Home") { e.preventDefault(); s.moveTo(id, 0); requestAnimationFrame(() => this.focusTile(id)); this._announce(`${id} moved to first`); return; }
    if (mod && e.key === "End") { e.preventDefault(); s.moveTo(id, s.order.length - 1); requestAnimationFrame(() => this.focusTile(id)); this._announce(`${id} moved to last`); return; }

    switch (e.key) {
      case "ArrowRight": e.preventDefault(); focusAt(Math.min(idx + 1, s.order.length - 1)); break;
      case "ArrowLeft": e.preventDefault(); focusAt(Math.max(idx - 1, 0)); break;
      case "ArrowDown": e.preventDefault(); focusAt(Math.min(idx + cols, s.order.length - 1)); break;
      case "ArrowUp": e.preventDefault(); focusAt(Math.max(idx - cols, 0)); break;
      case "Home": e.preventDefault(); focusAt(0); break;
      case "End": e.preventDefault(); focusAt(s.order.length - 1); break;
      case " ":
      case "Enter": e.preventDefault(); s.toggleSelect(id); this._anchor = id; this._setActive(id); break;
      case "Delete":
      case "Backspace": {
        e.preventDefault();
        const ids = s.selection.size ? [...s.selection] : [id];
        s.removeQueries(ids); this._announce(`${ids.length} removed`);
        break;
      }
      default: break;
    }
  }

  _announce(msg) { if (this.liveEl) this.liveEl.textContent = msg; }
}

function cssEscape(s) {
  if (window.CSS && CSS.escape) return CSS.escape(s);
  return String(s).replace(/[^a-zA-Z0-9_-]/g, "\\$&");
}
