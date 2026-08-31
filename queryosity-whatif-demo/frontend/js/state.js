// Schedule editing state for one workload: the current order (a permutation of
// a chosen subset of the workload's queries), a selection for batch operations,
// an undo/redo history over the order, and a saved baseline to compare against.
//
// Scores are NOT stored here -- the Scorer caches results by (workload, cap,
// ids), so re-scoring the baseline order is effectively free. State stays about
// structure; scoring stays in the workspace.

export class ScheduleState {
  constructor(workload, allIds) {
    this.workload = workload;
    this.allIds = allIds.slice(); // every query available in this workload
    this.original = allIds.slice(); // immutable full natural order
    this.order = allIds.slice(); // current schedule
    this.baseline = { order: allIds.slice(), label: "Original order" };
    this.selection = new Set();
    this._undo = [];
    this._redo = [];
    this._listeners = new Set();
  }

  // -- subscription --------------------------------------------------------
  subscribe(fn) {
    this._listeners.add(fn);
    return () => this._listeners.delete(fn);
  }
  _emit(reason) {
    for (const fn of this._listeners) fn(reason);
  }

  // -- history -------------------------------------------------------------
  _snapshot() {
    this._undo.push(this.order.slice());
    if (this._undo.length > 200) this._undo.shift();
    this._redo.length = 0;
  }
  canUndo() {
    return this._undo.length > 0;
  }
  canRedo() {
    return this._redo.length > 0;
  }
  undo() {
    if (!this._undo.length) return;
    this._redo.push(this.order.slice());
    this.order = this._undo.pop();
    this._pruneSelection();
    this._emit("undo");
  }
  redo() {
    if (!this._redo.length) return;
    this._undo.push(this.order.slice());
    this.order = this._redo.pop();
    this._pruneSelection();
    this._emit("redo");
  }

  // -- queries in / out of the schedule -----------------------------------
  inSchedule(id) {
    return this.order.includes(id);
  }
  missingIds() {
    const inSched = new Set(this.order);
    return this.allIds.filter((id) => !inSched.has(id));
  }

  setOrder(newOrder, reason = "set") {
    this._snapshot();
    this.order = newOrder.slice();
    this._pruneSelection();
    this._emit(reason);
  }

  addQueries(ids) {
    const have = new Set(this.order);
    const add = ids.filter((id) => this.allIds.includes(id) && !have.has(id));
    if (!add.length) return;
    this._snapshot();
    this.order = this.order.concat(add);
    this._emit("add");
  }

  removeQueries(ids) {
    const drop = new Set(ids);
    if (![...drop].some((id) => this.order.includes(id))) return;
    this._snapshot();
    this.order = this.order.filter((id) => !drop.has(id));
    this._pruneSelection();
    this._emit("remove");
  }

  // Keep only the given ids (in their current relative order); drop the rest.
  keepOnly(ids) {
    const keep = new Set(ids);
    const next = this.order.filter((id) => keep.has(id));
    if (next.length === this.order.length) return;
    this._snapshot();
    this.order = next;
    this._pruneSelection();
    this._emit("keep-only");
  }

  // Insert missing ids immediately after `afterId` (or at the end if not found).
  insertAfter(ids, afterId) {
    const have = new Set(this.order);
    const add = ids.filter((id) => this.allIds.includes(id) && !have.has(id));
    if (!add.length) return;
    this._snapshot();
    const at = this.order.indexOf(afterId);
    const next = this.order.slice();
    next.splice(at < 0 ? next.length : at + 1, 0, ...add);
    this.order = next;
    this._emit("add");
  }

  // -- reordering ----------------------------------------------------------
  moveTo(id, toIndex) {
    const from = this.order.indexOf(id);
    if (from < 0) return;
    const clamped = Math.max(0, Math.min(toIndex, this.order.length - 1));
    if (from === clamped) return;
    this._snapshot();
    const next = this.order.slice();
    next.splice(from, 1);
    next.splice(clamped, 0, id);
    this.order = next;
    this._emit("move");
  }

  moveBy(id, delta) {
    const from = this.order.indexOf(id);
    if (from < 0) return;
    this.moveTo(id, from + delta);
  }

  // Move a batch of ids (preserving their relative order) to sit before the
  // item currently at `beforeIndex` (or to the end if beforeIndex is null).
  moveBatch(ids, beforeIndex) {
    const set = new Set(ids);
    const picked = this.order.filter((id) => set.has(id));
    if (!picked.length) return;
    this._snapshot();
    const rest = this.order.filter((id) => !set.has(id));
    let idx = beforeIndex === null || beforeIndex === undefined ? rest.length : beforeIndex;
    idx = Math.max(0, Math.min(idx, rest.length));
    const next = rest.slice(0, idx).concat(picked, rest.slice(idx));
    this.order = next;
    this._emit("move-batch");
  }

  reverse() {
    this._snapshot();
    this.order = this.order.slice().reverse();
    this._emit("reverse");
  }

  randomize(rng = Math.random) {
    this._snapshot();
    const a = this.order.slice();
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    this.order = a;
    this._emit("randomize");
  }

  restoreOriginal() {
    this._snapshot();
    this.order = this.original.slice();
    this._pruneSelection();
    this._emit("restore");
  }

  loadOrder(order, reason = "load") {
    // Keep only ids that exist in the workload; append nothing.
    const known = new Set(this.allIds);
    const clean = order.filter((id) => known.has(id));
    this._snapshot();
    this.order = clean;
    this._pruneSelection();
    this._emit(reason);
  }

  // -- baseline ------------------------------------------------------------
  saveBaseline(label = "Saved baseline") {
    this.baseline = { order: this.order.slice(), label };
    this._emit("baseline");
  }
  resetBaselineToOriginal() {
    this.baseline = { order: this.original.slice(), label: "Original order" };
    this._emit("baseline");
  }

  // -- selection -----------------------------------------------------------
  toggleSelect(id) {
    if (this.selection.has(id)) this.selection.delete(id);
    else this.selection.add(id);
    this._emit("selection");
  }
  selectOnly(id) {
    this.selection = new Set([id]);
    this._emit("selection");
  }
  selectAll() {
    this.selection = new Set(this.order);
    this._emit("selection");
  }
  clearSelection() {
    if (!this.selection.size) return;
    this.selection.clear();
    this._emit("selection");
  }
  selectRangeTo(id, anchorId) {
    const a = this.order.indexOf(anchorId);
    const b = this.order.indexOf(id);
    if (a < 0 || b < 0) return;
    const [lo, hi] = a < b ? [a, b] : [b, a];
    for (let i = lo; i <= hi; i++) this.selection.add(this.order[i]);
    this._emit("selection");
  }
  _pruneSelection() {
    const inSched = new Set(this.order);
    for (const id of [...this.selection]) if (!inSched.has(id)) this.selection.delete(id);
  }

  // -- import / export -----------------------------------------------------
  exportObject(capacityPages) {
    return {
      kind: "queryosity-whatif-schedule",
      version: 1,
      workload: this.workload,
      capacity_pages: capacityPages,
      order: this.order.slice(),
    };
  }
}
