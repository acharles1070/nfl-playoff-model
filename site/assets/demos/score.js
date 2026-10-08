import { h, s, loadJSON, plot, responsive, f4, signed, segmented, pct } from "../lib.js";

export async function mount(el) {
  const d = await loadJSON("scorecard");
  let metric = "log_loss";
  el.append(h("h3", {}, "📋 Every model, same 2,494 games"), h("p", { class: "sub" }, "Holdout 2017–2025, every game that had a betting line."));
  const seg = segmented([["log_loss", "Log loss (lower = better)"], ["brier", "Brier (lower = better)"], ["accuracy", "Pick accuracy (higher = better)"]], metric, (v) => { metric = v; draw(); });
  const box = h("div", { class: "chart" });
  const note = h("div", { class: "sub", style: { marginTop: "10px" } });
  el.append(seg, box, note);
  const colors = { p_home_only: "var(--muted)", p_std_epa: "var(--blue)", p_kalman: "var(--model)", p_champion: "var(--model)", p_market: "var(--market)" };
  const market = d.models.find((m) => m.key === "p_market");

  function draw() {
    const vals = d.models.map((m) => m[metric]);
    const lo = metric === "accuracy" ? 0.5 : Math.min(...vals) - (metric === "brier" ? 0.01 : 0.01);
    const hi = metric === "accuracy" ? Math.max(...vals) + 0.02 : Math.max(...vals) + 0.01;
    responsive(box, (w) => {
      const narrow = w < 520, rowH = 46, left = narrow ? 86 : 190;
      const SHORT = { p_home_only: "Home field", p_std_epa: "Season EPA", p_kalman: "Kalman", p_champion: "Kalman + QB", p_market: "Market" };
      const { svg, g, x } = plot(w, d.models.length * rowH + 40, { xDomain: [lo, hi], yDomain: [0, 1], yTicks: [], margin: { left, top: 6, right: 64 }, xFmt: (v) => (metric === "accuracy" ? pct(v, 0) : v.toFixed(3)), grid: false });
      d.models.forEach((m, i) => {
        const cy = i * rowH + rowH / 2, c = colors[m.key], v = m[metric];
        g.append(s("text", { x: -10, y: cy + 4, "text-anchor": "end", style: { fill: "var(--text)", fontWeight: m.key === "p_champion" ? 800 : 500 } }, narrow ? SHORT[m.key] : m.label));
        g.append(s("rect", { x: x(lo), y: cy - 11, width: Math.max(2, x(v) - x(lo)), height: 22, rx: 5, fill: c, opacity: m.key === "p_champion" ? 1 : 0.75 }));
        const reach = metric === "log_loss" && m.key !== "p_market" ? x(market.log_loss + m.gap_hi) : x(v);
        g.append(s("text", { x: Math.max(x(v), reach) + 10, y: cy + 4, style: { fill: "var(--text)", fontFamily: "var(--mono)" } }, metric === "accuracy" ? pct(v) : f4(v)));
        if (metric === "log_loss" && m.key !== "p_market") {
          const a = x(market.log_loss + m.gap_lo), b = x(market.log_loss + m.gap_hi);
          g.append(s("line", { x1: a, x2: b, y1: cy, y2: cy, stroke: "var(--text)", "stroke-width": 2 }), s("line", { x1: a, x2: a, y1: cy - 6, y2: cy + 6, stroke: "var(--text)", "stroke-width": 2 }), s("line", { x1: b, x2: b, y1: cy - 6, y2: cy + 6, stroke: "var(--text)", "stroke-width": 2 }));
        }
      });
      if (metric === "log_loss") g.append(s("line", { x1: x(market.log_loss), x2: x(market.log_loss), y1: 0, y2: d.models.length * rowH, stroke: "var(--market)", "stroke-dasharray": "4 4" }));
      return svg;
    });
    const champ = d.models.find((m) => m.key === "p_champion");
    note.textContent = metric === "log_loss"
      ? `Black whiskers: 95% range of each model's gap to the market (champion: ${signed(champ.gap_vs_market)} [${signed(champ.gap_lo)}, ${signed(champ.gap_hi)}]). The dashed line is the market. None of the whiskers reach it.`
      : metric === "brier" ? "Brier score is the average squared miss. It tells the same story."
      : `Picking the favorite by my probabilities gets ${pct(champ.accuracy)} right versus ${pct(market.accuracy)} for the market. Accuracy ignores confidence, which is why I prefer log loss.`;
  }
  draw();
}
