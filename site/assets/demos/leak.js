import { h, loadJSON, f4, segmented } from "../lib.js";

export async function mount(el) {
  const d = await loadJSON("leakage");
  let peek = true;
  const lo = 0.55, hi = 0.70;
  const body = h("div", {});
  const note = h("div", { class: "caution", style: { marginTop: "14px" } });

  el.append(h("h3", {}, "🕵️ The peeking trap"), h("p", { class: "sub" }, `Same ${d.games.toLocaleString()} regular-season games (2017–2025), scored by log loss. Shorter bar = better forecast.`),
    segmented([["peek", "Rate teams with full-season stats (peeking)"], ["honest", "Only use games played before kickoff (honest)"]], "peek", (v) => { peek = v === "peek"; draw(); }), body, note);

  function draw() {
    const market = d.rows.find((r) => r.kind === "market").log_loss;
    body.replaceChildren(...d.rows.map((r) => {
      const fake = r.kind === "peek";
      const dim = fake && !peek;
      return h("div", { style: { margin: "14px 0", opacity: dim ? 0.35 : 1, transition: "opacity .4s" } },
        h("div", { class: "row", style: { justifyContent: "space-between", flexWrap: "nowrap" } }, h("span", { style: { fontWeight: 700 } }, r.label, fake ? h("span", { class: "pill bad", style: { marginLeft: "8px" } }, "uses the future") : null), h("span", { class: "mono" }, f4(r.log_loss))),
        h("div", { style: { background: "var(--panel2)", borderRadius: "8px", height: "18px", overflow: "hidden" } },
          h("div", { style: { width: `${Math.max(2, ((r.log_loss - lo) / (hi - lo)) * 100)}%`, height: "100%", borderRadius: "8px", transition: "width .5s", background: r.kind === "market" ? "var(--market)" : fake ? "var(--bad)" : "var(--model)" } })));
    }));
    const peekRow = d.rows.find((r) => r.kind === "peek");
    note.replaceChildren(h("b", { class: "k" }, peek ? "Looks amazing, doesn't it?" : "The honest picture"),
      peek ? `Rating teams by full-season EPA scores ${f4(peekRow.log_loss)}, better than the betting market (${f4(market)}). But each “prediction” already contains the game it's predicting. It's an answer key, not a forecast.`
           : `Using only what was known before kickoff, the same idea drops to ${f4(d.rows.find((r) => r.label.includes("season-to-date")).log_loss)}, a long way behind the market. The ${f4(d.rows.find((r) => r.label.includes("season-to-date")).log_loss - peekRow.log_loss)} of “skill” in the peeking version was never real.`);
  }
  draw();
}
