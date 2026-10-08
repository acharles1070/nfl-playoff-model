import { h, loadJSON } from "../lib.js";
import { forest } from "../forest.js";

export async function mount(el) {
  const reg = await loadJSON("registry");
  const rows = reg.rows.filter((r) => ["E02", "E03", "E04", "E05"].includes(r.id));
  el.append(h("h3", {}, "Does tracking the quarterback help?"), h("p", { class: "sub" }, "Change in log loss from adding the QB layer (left of the dashed line = better). Bars show 95% ranges. Amber = looked significant on its own; none survive correction for the many ideas I tried."));
  const box = h("div", { class: "chart" });
  el.append(box);
  forest(box, rows, {
    tip: (r) => `<b>${r.name}</b><br>change ${r.delta.toFixed(4)} [${r.lo.toFixed(4)}, ${r.hi.toFixed(4)}]<br>p = ${r.p.toFixed(3)}`,
    domain: [-0.03, 0.03], labelW: 0.4,
  });
}
