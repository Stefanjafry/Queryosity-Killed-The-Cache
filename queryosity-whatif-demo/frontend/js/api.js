// Thin API seam. Generated research schedules are retrieved as prepared
// artifacts; arbitrary edited orders go only to POST /api/score.

async function getJSON(url) {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw apiError(body, res.status);
  return body;
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "application/json" },
    body: JSON.stringify(payload),
  });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw apiError(body, res.status);
  return body;
}

function apiError(body, status) {
  const info = (body && body.error) || {};
  const err = new Error(info.message || `request failed (${status})`);
  err.code = info.code || "http_error";
  err.status = status;
  err.detail = info.detail || null;
  return err;
}

const qs = (params) => new URLSearchParams(params).toString();

export const Api = {
  health: () => getJSON("/api/health"),
  workloads: () => getJSON("/api/workloads"),
  generated: (workload, cap) => getJSON(`/api/generated-schedules?${qs({ workload, cap })}`),
  explanation: (workload, cap, method) =>
    getJSON(`/api/explanation?${qs({ workload, cap, method })}`),
  // compatibility with editor code from the previous demo
  schedulersFor: (workload, cap) => getJSON(`/api/generated-schedules?${qs({ workload, cap })}`),
  validate: (workload, order) => postJSON("/api/schedules/validate", { workload, order }),
  scoreRaw: (workload, cap, order) => postJSON("/api/score", { workload, cap, order }),
  compare: (workload, cap, schedules) => postJSON("/api/compare", { workload, cap, schedules }),
  optimizeSubset: (workload, cap, order) => postJSON("/api/optimize", { workload, cap, order }),
};

function keyFor(workload, cap, order) {
  return `${workload}|${cap}|${order.join(",")}`;
}

export class Scorer {
  constructor() {
    this._cache = new Map();
    this._seq = 0;
    this._latestApplied = 0;
  }

  cached(workload, cap, order) {
    return this._cache.get(keyFor(workload, cap, order)) || null;
  }

  async get(workload, cap, order) {
    const key = keyFor(workload, cap, order);
    const hit = this._cache.get(key);
    if (hit) return { result: hit, fromCache: true };
    const result = await Api.scoreRaw(workload, cap, order);
    this._cache.set(key, result);
    return { result, fromCache: false };
  }

  async score(workload, cap, order) {
    const key = keyFor(workload, cap, order);
    const myId = ++this._seq;
    const hit = this._cache.get(key);
    if (hit) {
      const stale = myId < this._latestApplied;
      if (!stale) this._latestApplied = myId;
      return { result: hit, fromCache: true, stale };
    }
    const result = await Api.scoreRaw(workload, cap, order);
    this._cache.set(key, result);
    const stale = myId < this._latestApplied;
    if (!stale) this._latestApplied = myId;
    return { result, fromCache: false, stale };
  }
}

export function debounce(fn, wait) {
  let t = null;
  const wrapped = (...args) => {
    if (t) clearTimeout(t);
    t = setTimeout(() => { t = null; fn(...args); }, wait);
  };
  wrapped.cancel = () => { if (t) clearTimeout(t); t = null; };
  return wrapped;
}
