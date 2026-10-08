// Forest plot: one row per estimate with a 95% interval, zero line, and optional highlight.
import { s, h, plot, responsive, tooltip, linear, niceTicks, signed } from "./lib.js";

export function forest(container, rows, { rowH = 26, label = (r) => r.name, color = (r) => (r.survives ? "var(--accent)" : r.raw_sig ? "var(--warn)" : "var(--muted)"), tip, labelW = 0.46, domain } = {}) {
  const tt = tooltip(container);
  const lo = domain ? domain[0] : Math.min(...rows.map((r) => r.lo));
  const hi = domain ? domain[1] : Math.max(...rows.map((r) => r.hi));
  const pad = (hi - lo) * 0.06;

  return responsive(container, (w) => {
    const narrow = w < 560;
    const left = narrow ? 8 : Math.round(w * labelW);
    const rh = narrow ? 44 : rowH;
    const height = rows.length * rh + 44;
    const { svg, g, x, iw } = plot(w, height, {
      xDomain: [lo - pad, hi + pad], yDomain: [0, rows.length], margin: { left, bottom: 34, top: 6, right: 10 },
      yTicks: [], xFmt: (v) => v.toFixed(3), xLabel: "change in log loss (negative = better)",
    });
    g.append(s("line", { x1: x(0), x2: x(0), y1: 0, y2: rows.length * rh, stroke: "var(--text)", "stroke-width": 1.2, "stroke-dasharray": "4 3", opacity: 0.8 }));

    rows.forEach((r, i) => {
      const cy = narrow ? i * rh + 32 : i * rh + rh / 2;
      const c = color(r);
      const grp = s("g", {},
        s("rect", { x: -left, y: i * rh, width: left + iw + 10, height: rh, fill: "transparent" }),
        narrow
          ? s("text", { x: 0, y: i * rh + 14, style: { fill: "var(--text)", fontSize: "11.5px", paintOrder: "stroke", stroke: "var(--panel)", strokeWidth: "5px" }, text: truncate(label(r), Math.floor(iw / 6.2)) })
          : s("text", { x: -8, y: cy + 4, "text-anchor": "end", style: { fill: "var(--text)", fontSize: "12px" }, text: truncate(label(r), 60) }),
        s("line", { x1: x(r.lo), x2: x(r.hi), y1: cy, y2: cy, stroke: c, "stroke-width": 2.5, "stroke-linecap": "round" }),
        s("circle", { cx: x(r.delta), cy, r: 4.5, fill: c }),
      );
      grp.addEventListener("pointermove", (ev) => tt.show(tip ? tip(r) : `<b>${r.name}</b><br>${signed(r.delta)} [${signed(r.lo)}, ${signed(r.hi)}]`, ev));
      grp.addEventListener("pointerleave", () => tt.hide());
      g.append(grp);
    });
    return svg;
  });
}

function truncate(t, n) { return t.length > n ? `${t.slice(0, n - 1)}…` : t; }
