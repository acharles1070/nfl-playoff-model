import { h, s, loadJSON, plot, responsive, segmented, f2, pct, signed } from "../lib.js";

export async function mount(el) {
  const d = await loadJSON("betting");
  let th = "0.0", mode = "flat", base = true;
  el.append(h("h3", {}, "💸 Betting the model's picks (2017–2025)"), h("p", { class: "sub" }, "I bet whichever side the model thinks is underpriced, at the real closing price (with the bookmaker's margin). How big a disagreement before I bet?"));
  const thSeg = segmented(Object.keys(d.series).map((k) => [k, k === "0.0" ? "any edge" : `≥ ${Math.round(k * 100)} pts`]), th, (v) => { th = v; draw(); });
  const modeSeg = segmented([["flat", "Flat: 1 unit per bet"], ["kelly", "Quarter-Kelly: bankroll grows/shrinks"]], mode, (v) => { mode = v; draw(); });
  const box = h("div", { class: "chart" }), read = h("div", { class: "found", style: { marginTop: "10px" } });
  el.append(h("div", { class: "row" }, thSeg, modeSeg), box, read);

  function draw() {
    const ser = d.series[th];
    const curve = mode === "flat" ? ser.flat : ser.kelly;
    const bases = mode === "flat" ? Object.entries(d.baselines) : [];
    const ys = [...curve, ...bases.flatMap(([, v]) => v.curve)];
    const lo = Math.min(...ys, mode === "flat" ? 0 : 0), hi = Math.max(...ys, mode === "flat" ? 10 : 110);
    const n = Math.max(curve.length, ...bases.map(([, v]) => v.curve.length));

    responsive(box, (w) => {
      const { svg, g, x, y } = plot(w, 300, { xDomain: [0, n], yDomain: [lo, hi], xLabel: "bets placed, in date order", yLabel: mode === "flat" ? "profit (units)" : "bankroll (start = 100)", xFmt: (v) => v.toLocaleString(), margin: { left: 56 } });
      const line = (arr, color, wd, dash) => s("path", { d: arr.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(""), fill: "none", stroke: color, "stroke-width": wd, "stroke-dasharray": dash });
      g.append(s("line", { x1: 0, x2: x(n), y1: y(mode === "flat" ? 0 : 100), y2: y(mode === "flat" ? 0 : 100), stroke: "var(--muted)", opacity: 0.6 }));
      ser.season_end.forEach((i, k) => { g.append(s("line", { x1: x(i), x2: x(i), y1: 0, y2: y(lo), stroke: "var(--line)" })); });
      bases.forEach(([name, v], k) => g.append(line(v.curve, k ? "var(--blue)" : "var(--market)", 1.8, "5 4")));
      g.append(line(curve, "var(--bad)", 3));
      return svg;
    });

    const last = curve[curve.length - 1];
    read.replaceChildren(h("b", { class: "k" }, `${ser.bets.toLocaleString()} bets`),
      mode === "flat" ? `Result: ${last >= 0 ? "+" : "−"}${Math.abs(last).toFixed(0)} units, an average return of ${(ser.roi * 100).toFixed(1)}% per bet. For comparison, betting every favorite returns ${(d.baselines["Bet every favorite"].roi * 100).toFixed(1)}% and every home team ${(d.baselines["Bet every home team"].roi * 100).toFixed(1)}%.`
                     : `Result: a bankroll of 100 ends at ${last.toFixed(1)}. Staking in proportion to “edge” makes sense only if the probabilities are right; when they're wrong, it compounds the losses.`);
  }
  el.append(h("div", { class: "legend" }, ...[["var(--bad)", "model's picks"], ["var(--market)", "bet every favorite (flat only)"], ["var(--blue)", "bet every home team (flat only)"]].map(([c, t]) => h("span", {}, h("i", { style: { background: c } }), t)), h("span", {}, "Vertical lines separate seasons 2017 → 2025")));
  draw();
}
