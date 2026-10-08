import { h, s, plot, responsive, normCdf, gaussian, mulberry32, loadJSON } from "../lib.js";

let M = 37;               // overwritten with the registry's count at mount
const ALPHA = 0.05;

function pvals(rng) { return Array.from({ length: M }, () => 2 * (1 - normCdf(Math.abs(gaussian(rng))))); }

function holm(ps) {
  const order = ps.map((p, i) => [p, i]).sort((a, b) => a[0] - b[0]);
  const keep = new Set();
  for (let k = 0; k < order.length; k++) {
    if (order[k][0] <= ALPHA / (M - k)) keep.add(order[k][1]); else break;
  }
  return keep;
}

export async function mount(el) {
  M = (await loadJSON("registry")).n;
  const rng = mulberry32(99);
  let runs = [], last = null;
  el.append(h("h3", {}, "🎰 The luck simulator"), h("p", { class: "sub" }, `Each press runs ${M} experiments where nothing is real (every idea is useless). Dots that fall in the shaded zone look “significant.” How many do you get?`));
  const btns = h("div", { class: "row" }, h("button", { class: "btn primary", onclick: () => { step(1); } }, `Run ${M} useless experiments`), h("button", { class: "btn", onclick: () => step(1000) }, "Run it 1,000 times"), h("button", { class: "btn", onclick: () => { runs = []; last = null; draw(); } }, "Reset"));
  const box = h("div", { class: "chart" }), read = h("div", { class: "found", style: { marginTop: "10px" } }), hist = h("div", { class: "chart" });
  el.append(btns, box, read, hist);

  function step(n) {
    for (let k = 0; k < n; k++) {
      const ps = pvals(rng), sig = ps.filter((p) => p < ALPHA).length;
      last = { ps, sig, holm: holm(ps).size };
      runs.push({ sig: last.sig, holm: last.holm });
    }
    draw();
  }

  function draw() {
    responsive(box, (w) => {
      const { svg, g, x } = plot(w, 120, { xDomain: [0, 1], yDomain: [0, 1], yTicks: [], xFmt: (v) => v.toFixed(1), xLabel: "p-value of each useless experiment (smaller = “more significant”)", margin: { left: 14, top: 6 }, grid: false });
      g.append(s("rect", { x: x(0), y: 0, width: x(ALPHA) - x(0), height: 62, fill: "var(--bad)", opacity: 0.2 }));
      g.append(s("text", { x: x(ALPHA) + 4, y: 14, style: { fill: "var(--bad)" }, text: "“significant” (p < 0.05)" }));
      if (last) {
        const rng2 = mulberry32(5);
        last.ps.forEach((p) => g.append(s("circle", { cx: x(p), cy: 22 + rng2() * 38, r: 5.5, fill: p < ALPHA ? "var(--bad)" : "var(--muted)", opacity: 0.9 })));
      }
      return svg;
    });
    const n = runs.length;
    const anyRaw = n ? runs.filter((r) => r.sig > 0).length / n : 0;
    const anyHolm = n ? runs.filter((r) => r.holm > 0).length / n : 0;
    read.replaceChildren(h("b", { class: "k" }, n ? `${n.toLocaleString()} run${n === 1 ? "" : "s"} so far` : "Press a button"),
      last ? `Last run: ${last.sig} useless idea${last.sig === 1 ? "" : "s"} looked “significant.” After the Holm correction: ${last.holm}. ` : "",
      n > 1 ? `Across all runs, ${(anyRaw * 100).toFixed(0)}% had at least one fake “discovery” (theory: ${((1 - 0.95 ** M) * 100).toFixed(0)}%), versus ${(anyHolm * 100).toFixed(1)}% after correction.` : "");
    responsive(hist, (w) => {
      if (n < 20) return h("span", {});
      const max = 10, counts = Array(max + 1).fill(0);
      runs.forEach((r) => (counts[Math.min(r.sig, max)] += 1));
      const top = Math.max(...counts) / n;
      const { svg, g, x, y } = plot(w, 150, { xDomain: [-0.5, max + 0.5], yDomain: [0, top * 1.1], xTicks: [...Array(max + 1).keys()], xFmt: (v) => (v === max ? `${max}+` : v), yFmt: (v) => `${Math.round(v * 100)}%`, xLabel: "number of fake “discoveries” in one run of 38", yLabel: "share of runs", margin: { left: 56 } });
      counts.forEach((c, k) => g.append(s("rect", { x: x(k - 0.4), width: x(k + 0.4) - x(k - 0.4), y: y(c / n), height: y(0) - y(c / n), fill: "var(--bad)", opacity: 0.75, rx: 2 })));
      g.append(s("line", { x1: x(M * ALPHA), x2: x(M * ALPHA), y1: 0, y2: y(0), stroke: "var(--text)", "stroke-dasharray": "4 3" }), s("text", { x: x(M * ALPHA) + 5, y: 12, text: "expected: 1.9" }));
      return svg;
    });
  }
  draw();
}
