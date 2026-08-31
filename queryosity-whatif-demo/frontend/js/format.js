// Number / byte / percentage formatting. Kept tiny and dependency-free.

const NF = new Intl.NumberFormat("en-US");

export function int(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "\u2014";
  return NF.format(Math.round(n));
}

// Compact figures for dense readouts: 1.23M, 456K, 789.
export function compact(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "\u2014";
  const a = Math.abs(n);
  if (a >= 1e9) return trim(n / 1e9) + "B";
  if (a >= 1e6) return trim(n / 1e6) + "M";
  if (a >= 1e4) return trim(n / 1e3) + "K";
  return int(n);
}

function trim(x) {
  return (Math.round(x * 100) / 100).toString();
}

export function pct(ratio, digits = 1) {
  if (ratio === null || ratio === undefined || Number.isNaN(ratio)) return "\u2014";
  return (ratio * 100).toFixed(digits) + "%";
}

// Percentage-points delta, always signed.
export function pp(deltaRatio, digits = 1) {
  if (deltaRatio === null || deltaRatio === undefined || Number.isNaN(deltaRatio))
    return "\u2014";
  const v = deltaRatio * 100;
  const s = v.toFixed(digits);
  return (v > 0 ? "+" : "") + s + " pp";
}

export function signed(n) {
  if (n === null || n === undefined || Number.isNaN(n)) return "\u2014";
  const r = Math.round(n);
  return (r > 0 ? "+" : "") + int(r);
}

export function bytes(num) {
  if (num === null || num === undefined || Number.isNaN(num)) return "\u2014";
  const step = 1024;
  let val = num;
  const units = ["B", "KB", "MB", "GB", "TB"];
  for (let i = 0; i < units.length; i++) {
    if (val < step || i === units.length - 1) {
      if (units[i] === "B" || units[i] === "KB") return `${Math.round(val)} ${units[i]}`;
      const s = (Math.round(val * 100) / 100).toString();
      return `${s} ${units[i]}`;
    }
    val /= step;
  }
  return `${num} B`;
}

export function ms(x) {
  if (x === null || x === undefined || Number.isNaN(x)) return "\u2014";
  if (x < 1) return "<1 ms";
  if (x < 1000) return `${Math.round(x)} ms`;
  return `${(x / 1000).toFixed(2)} s`;
}

// Plain-language description of a baseline -> modified change in hit behaviour.
export function describeDelta(base, mod) {
  if (!base || !mod) return "";
  const dH = mod.total_hits - base.total_hits;
  const dRatio = mod.hit_ratio - base.hit_ratio;
  const dir = dH === 0 ? "left unchanged" : dH > 0 ? "increased" : "decreased";
  if (dH === 0) {
    return `This reordering left the simulated hit ratio unchanged at ${pct(mod.hit_ratio)}.`;
  }
  const hitsWord = dH > 0 ? "more" : "fewer";
  return (
    `This reordering ${dir} the simulated hit ratio ` +
    `${pct(base.hit_ratio)} \u2192 ${pct(mod.hit_ratio)} (${pp(dRatio)}), ` +
    `about ${int(Math.abs(dH))} ${hitsWord} theoretical buffer hits ` +
    `out of ${int(mod.total_requests)} page requests.`
  );
}
