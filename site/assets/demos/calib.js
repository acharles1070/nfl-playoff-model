import { h, s, loadJSON, plot, responsive, tooltip, f3, pct, segmented } from "../lib.js";

export async function mount(el) {
  const d = await loadJSON("calibration");
  let which = "both";
  el.append(h("h3", {}, "🎚️ Do the probabilities mean what they say?"), h("p", { class: "sub" }, "Games grouped by the probability each forecaster gave the home team. If forecasts are calibrated, the dots sit on the diagonal: of games called 70%, about 70% happen."));
  const seg = segmented([["both", "Both"], ["p_champion", "My model"], ["p_market", "Market"]], which, (v) => { which = v; draw(); });
  const box = h("div", { class: "chart" }), note = h("div", { class: "sub", style: { marginTop: "8px" } });
  el.append(seg, box, note);
  const tt = tooltip(box);

  function draw() {
    responsive(box, (w) => {
      const size = Math.min(w, 520);
      const { svg, g, x, y } = plot(size, size * 0.8, { xDomain: [0, 1], yDomain: [0, 1], xFmt: (v) => `${Math.round(v * 100)}%`, yFmt: (v) => `${Math.round(v * 100)}%`, xLabel: "what the forecaster said", yLabel: "what actually happened", margin: { left: 46 } });
      g.append(s("line", { x1: x(0), y1: y(0), x2: x(1), y2: y(1), stroke: "var(--muted)", "stroke-dasharray": "5 4" }));
      for (const key of ["p_champion", "p_market"]) {
        if (which !== "both" && which !== key) continue;
        const color = key === "p_market" ? "var(--market)" : "var(--model)";
        d[key].bins.forEach((b) => {
          const dx = which === "both" ? (key === "p_market" ? 5 : -5) : 0;
          const grp = s("g", {},
            s("line", { x1: x(b.pred) + dx, x2: x(b.pred) + dx, y1: y(b.lo), y2: y(b.hi), stroke: color, "stroke-width": 2, opacity: 0.8 }),
            s("circle", { cx: x(b.pred) + dx, cy: y(b.obs), r: 4 + Math.sqrt(b.n) / 5, fill: color, opacity: 0.9 }));
          grp.addEventListener("pointermove", (ev) => tt.show(`<b>${key === "p_market" ? "Market" : "Model"}</b><br>said ${pct(b.pred, 0)} → happened ${pct(b.obs, 0)}<br>${b.n} games`, ev));
          grp.addEventListener("pointerleave", () => tt.hide());
          g.append(grp);
        });
      }
      return svg;
    });
    note.textContent = `Calibration slope (1.00 = perfect): model ${f3(d.p_champion.slope)}, market ${f3(d.p_market.slope)} on these ${"2,494"} games. Dot size = number of games; whiskers = 95% ranges. Both sit close to the diagonal; the market's edge is not miscalibration but sharper, better-informed probabilities.`;
  }
  draw();
}
