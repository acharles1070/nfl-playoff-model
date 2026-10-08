import { h, s, plot, responsive, normCdf, slider, f4, loadJSON } from "../lib.js";

const PER_SEASON = 285;   // NFL games per season (the registry uses the same figure)

export async function mount(el) {
  const SD = (await loadJSON("power")).sigma;   // paired per-game log-loss SD, from the registry
  const power = (delta, n) => { const z = delta / (SD / Math.sqrt(n)); return normCdf(z - 1.96) + normCdf(-z - 1.96); };
  const needed = (delta) => Math.ceil(((1.96 + 0.8416) * SD / delta) ** 2);
  let delta = 0.005, logn = Math.log10(2494);
  el.append(h("h3", {}, "🔭 Could the test even see it?"), h("p", { class: "sub" }, "Per-game log-loss differences between two similar models vary by about ±0.100. Averaging over more games shrinks the noise."));
  const dCtl = slider({ label: "True improvement to detect (log loss)", min: 0.001, max: 0.03, step: 0.0005, value: delta, fmt: (v) => v.toFixed(4), onInput: (v) => { delta = v; draw(); } });
  const nCtl = slider({ label: "Number of games tested", min: 1.7, max: 4.5, step: 0.01, value: logn, fmt: (v) => Math.round(10 ** v).toLocaleString(), onInput: (v) => { logn = v; draw(); } });
  const box = h("div", { class: "chart" }), read = h("div", { class: "found", style: { marginTop: "10px" } });
  el.append(h("div", { class: "row" }, dCtl.el, nCtl.el), box, read);

  function draw() {
    const n = Math.round(10 ** logn);
    responsive(box, (w) => {
      const { svg, g, x, y } = plot(w, 260, { xDomain: [1.7, 4.5], yDomain: [0, 1], xTicks: [2, 3, 4], xFmt: (v) => (10 ** v).toLocaleString(), yFmt: (v) => `${Math.round(v * 100)}%`, xLabel: "games tested (log scale)", yLabel: "chance of detecting", margin: { left: 48 } });
      const pts = Array.from({ length: 120 }, (_, k) => { const lx = 1.7 + (k * 2.8) / 119; return [lx, power(delta, 10 ** lx)]; });
      g.append(s("path", { d: pts.map(([a, b], i) => `${i ? "L" : "M"}${x(a).toFixed(1)},${y(b).toFixed(1)}`).join(""), fill: "none", stroke: "var(--model)", "stroke-width": 3 }));
      g.append(s("line", { x1: 0, x2: x(4.5), y1: y(0.8), y2: y(0.8), stroke: "var(--muted)", "stroke-dasharray": "4 4" }), s("text", { x: 6, y: y(0.8) - 5, text: "80% = a reliable test" }));
      [[111, "111 playoff games"], [2494, "2,494 holdout games"]].forEach(([k, lab], i) => {
        g.append(s("line", { x1: x(Math.log10(k)), x2: x(Math.log10(k)), y1: 0, y2: y(0), stroke: "var(--market)", "stroke-dasharray": "3 3" }), s("text", { x: x(Math.log10(k)) + 4, y: 14 + i * 14, style: { fill: "var(--market)" }, text: lab }));
      });
      g.append(s("circle", { cx: x(logn), cy: y(power(delta, n)), r: 7, fill: "var(--you)" }));
      return svg;
    });
    const p = power(delta, n), need = needed(delta);
    read.replaceChildren(h("b", { class: "k" }, `${n.toLocaleString()} games, true improvement ${delta.toFixed(4)}`),
      `A test would catch it ${Math.round(p * 100)}% of the time. A reliable (80%) test needs about ${need.toLocaleString()} games, roughly ${Math.max(1, Math.round(need / PER_SEASON)).toLocaleString()} NFL season${need / PER_SEASON < 1.5 ? "" : "s"}. `,
      `With only 111 playoff games, the smallest improvement a reliable test can see is about ${(2.8016 * SD / Math.sqrt(111)).toFixed(3)}; with 2,494 holdout games it is ${(2.8016 * SD / Math.sqrt(2494)).toFixed(4)}.`);
  }
  draw();
}
