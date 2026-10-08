import { h, loadJSON, segmented, signed } from "../lib.js";
import { forest } from "../forest.js";

export async function mount(el) {
  const reg = await loadJSON("registry");
  let filter = "all";
  el.append(h("h3", {}, `📚 All ${reg.n} experiments`), h("p", { class: "sub" }, "Change in log loss versus the model each idea was compared with. Left of the dashed line = better. Hover for details."));
  const seg = segmented([["all", `All ${reg.n}`], ["raw", `Looked significant (${reg.raw_sig})`], ["holm", `Survived correction (${reg.holm})`]], filter, (v) => { filter = v; draw(); });
  const box = h("div", { class: "chart" });
  el.append(seg, box, h("div", { class: "legend" }, ...[["var(--accent)", "survives Holm / Benjamini–Hochberg"], ["var(--warn)", "looked significant, did not survive"], ["var(--muted)", "no detectable effect"]].map(([c, t]) => h("span", {}, h("i", { style: { background: c } }), t))));

  function draw() {
    const rows = reg.rows.filter((r) => filter === "all" || (filter === "raw" ? r.raw_sig : r.survives));
    box.replaceChildren();
    forest(box, rows, { label: (r) => `${r.id} ${r.name}`, domain: [-0.03, 0.03], labelW: 0.5,
      tip: (r) => `<b>${r.id}: ${r.name}</b><br>change ${r.delta.toFixed(4)} [${r.lo.toFixed(4)}, ${r.hi.toFixed(4)}]<br>p = ${r.p.toFixed(3)}, after Holm: ${r.p_holm.toFixed(3)}` });
  }
  draw();

  // does the one survivor replicate on seasons that had no part in its design?
  const rep = (await loadJSON("replication")).rows;
  el.append(h("h3", { style: { marginTop: "22px" } }, "Does the survivor replicate?"),
    h("p", { class: "sub" }, "The Kalman filter was found on 2017–2025, when that was the only test window. I pre-specified a check on seasons 2003–2009, which had no part in its design or tuning (criterion: a 95% interval excluding zero)."),
    h("div", { class: "scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ...["Test seasons", "Games", "Change in log loss [95% CI]", "Seasons better"].map((t, i) => h("th", { class: i ? "num" : "" }, t)))),
      h("tbody", {}, ...rep.map((r) => h("tr", {}, h("td", {}, r.window.replace(/^(primary|secondary): /, (m) => (m.startsWith("primary") ? "★ primary: " : ""))), h("td", { class: "num" }, r.games.toLocaleString()),
        h("td", { class: "num" }, `${signed(r.delta_log_loss)} [${signed(r.ci_low)}, ${signed(r.ci_high)}]`), h("td", { class: "num" }, `${r.per_season_better}/${r.n_seasons}`)))))),
    h("div", { class: "caution" }, h("b", { class: "k" }, "Verdict"), "Same direction, about half the size, and the interval just touches zero. So the survivor is supported across windows, but not independently confirmed at the 5% level."));
}
