// Tiny helpers shared by every demo: DOM/SVG builders, data loading, scales, math.

export const $ = (sel, el = document) => el.querySelector(sel);
export const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];

const SVG_NS = "http://www.w3.org/2000/svg";

function apply(el, attrs) {
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.setAttribute("class", v);
    else if (k === "html") el.innerHTML = v;
    else if (k === "text") el.textContent = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
}

function append(el, kids) {
  for (const kid of kids.flat(Infinity)) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  }
  return el;
}

export function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  apply(el, attrs);
  return append(el, kids);
}

export function s(tag, attrs, ...kids) {
  const el = document.createElementNS(SVG_NS, tag);
  apply(el, attrs);
  return append(el, kids);
}

const cache = new Map();
export function loadJSON(name) {
  if (!cache.has(name)) cache.set(name, fetch(`data/${name}.json`, { cache: "no-cache" }).then((r) => {
    if (!r.ok) throw new Error(`Could not load data/${name}.json (${r.status})`);
    return r.json();
  }));
  return cache.get(name);
}

// ---- formatting
export const f2 = (x) => x.toFixed(2);
export const f3 = (x) => x.toFixed(3);
export const f4 = (x) => x.toFixed(4);
export const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;
export const signed = (x, d = 4) => `${x >= 0 ? "+" : "−"}${Math.abs(x).toFixed(d)}`;

// ---- scales and ticks
export function linear(d0, d1, r0, r1) {
  const k = (r1 - r0) / (d1 - d0 || 1);
  const fn = (x) => r0 + (x - d0) * k;
  fn.invert = (y) => d0 + (y - r0) / k;
  return fn;
}

export function niceTicks(lo, hi, n = 5) {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((v) => v >= raw) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

// ---- charts: redraw at the container's real pixel width so text stays readable
const renderers = new WeakMap();

export function responsive(container, draw) {
  const first = !renderers.has(container);
  renderers.set(container, draw);                       // a re-draw replaces the old drawing function
  const render = () => {
    const w = Math.max(280, Math.floor(container.clientWidth));
    container.replaceChildren(renderers.get(container)(w));
  };
  render();
  if (first && typeof ResizeObserver !== "undefined") {
    let last = container.clientWidth;
    new ResizeObserver(() => { if (Math.abs(container.clientWidth - last) > 4) { last = container.clientWidth; render(); } }).observe(container);
  }
  return render;
}

// A bare SVG with margins and optional axes; returns {svg, g, x, y, iw, ih}.
export function plot(width, height, { xDomain, yDomain, margin = {}, xTicks, yTicks, xFmt = String, yFmt = String, xLabel, yLabel, grid = true }) {
  const m = { top: 10, right: 14, bottom: 38, left: 52, ...margin };
  const iw = width - m.left - m.right;
  const ih = height - m.top - m.bottom;
  const x = linear(xDomain[0], xDomain[1], 0, iw);
  const y = linear(yDomain[0], yDomain[1], ih, 0);
  const svg = s("svg", { viewBox: `0 0 ${width} ${height}`, width, height, role: "img" });
  const g = s("g", { transform: `translate(${m.left},${m.top})` });
  svg.append(g);

  const xt = xTicks || niceTicks(xDomain[0], xDomain[1], Math.max(3, Math.floor(iw / 90)));
  const yt = yTicks || niceTicks(yDomain[0], yDomain[1], 5);

  if (grid) g.append(s("g", { class: "grid" }, ...yt.map((v) => s("line", { x1: 0, x2: iw, y1: y(v), y2: y(v) }))));
  g.append(
    s("g", { class: "axis" },
      s("line", { x1: 0, x2: iw, y1: ih, y2: ih }),
      ...xt.map((v) => s("g", { transform: `translate(${x(v)},${ih})` }, s("line", { y2: 5 }), s("text", { y: 19, "text-anchor": "middle", text: xFmt(v) }))),
      s("line", { x1: 0, x2: 0, y1: 0, y2: ih }),
      ...yt.map((v) => s("g", { transform: `translate(0,${y(v)})` }, s("line", { x2: -5 }), s("text", { x: -9, y: 4, "text-anchor": "end", text: yFmt(v) }))),
    ),
  );
  if (xLabel) g.append(s("text", { x: iw / 2, y: ih + 34, "text-anchor": "middle", text: xLabel }));
  if (yLabel) g.append(s("text", { transform: `translate(${-m.left + 12},${ih / 2}) rotate(-90)`, "text-anchor": "middle", text: yLabel }));
  return { svg, g, x, y, iw, ih };
}

export function pathFrom(points) {
  return points.map(([px, py], i) => `${i ? "L" : "M"}${px.toFixed(1)},${py.toFixed(1)}`).join("");
}

// A tooltip that follows the pointer inside a positioned container.
export function tooltip(container) {
  container.style.position = "relative";
  const tip = h("div", { style: { position: "absolute", pointerEvents: "none", display: "none", zIndex: 5, background: "var(--panel2)", border: "1px solid var(--line)", borderRadius: "8px", padding: "6px 9px", fontSize: ".82rem", boxShadow: "var(--shadow)", maxWidth: "260px" } });
  container.append(tip);
  return {
    show(html, ev) {
      const r = container.getBoundingClientRect();
      tip.innerHTML = html;
      tip.style.display = "block";
      const px = ev.clientX - r.left, py = ev.clientY - r.top;
      tip.style.left = `${Math.min(Math.max(8, px + 12), r.width - tip.offsetWidth - 8)}px`;
      tip.style.top = `${Math.max(4, py - tip.offsetHeight - 10)}px`;
    },
    hide() { tip.style.display = "none"; },
  };
}

// ---- math
export function normCdf(x) {
  // Abramowitz-Stegun 7.1.26 erf; absolute error < 1.5e-7
  const sign = x < 0 ? -1 : 1;
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return 0.5 * (1 + sign * y);
}

export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const logLoss1 = (p, y) => -(y ? Math.log(Math.max(p, 1e-9)) : Math.log(Math.max(1 - p, 1e-9)));

export function segmented(options, value, onChange) {
  const el = h("div", { class: "seg" });
  const paint = (v) => [...el.children].forEach((b) => b.classList.toggle("on", b.dataset.v === String(v)));
  options.forEach(([v, label]) => el.append(h("button", { type: "button", "data-v": v, onclick: () => { paint(v); onChange(v); } }, label)));
  paint(value);
  return el;
}

export function slider({ label, min, max, step, value, fmt = String, onInput }) {
  const out = h("b", {}, fmt(value));
  const input = h("input", { type: "range", min, max, step, value, "aria-label": label });
  input.addEventListener("input", () => { out.textContent = fmt(+input.value); onInput(+input.value); });
  return { el: h("div", { class: "ctl" }, h("label", {}, h("span", {}, label), out), input), input, out };
}
