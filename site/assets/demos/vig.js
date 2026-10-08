import { h, f3, pct } from "../lib.js";

const implied = (o) => (o < 0 ? -o / (-o + 100) : 100 / (o + 100));

export async function mount(el) {
  let a = -150, b = 130;
  el.append(h("h3", {}, "🧾 From betting price to probability"), h("p", { class: "sub" }, "American odds: −150 means risk 150 to win 100; +130 means risk 100 to win 130. Edit the numbers or pick a preset."));
  const inA = h("input", { type: "number", value: a, style: inputStyle(), "aria-label": "Home moneyline" });
  const inB = h("input", { type: "number", value: b, style: inputStyle(), "aria-label": "Away moneyline" });
  const out = h("div", { style: { marginTop: "12px" } });
  const upd = () => { a = +inA.value || -110; b = +inB.value || -110; draw(); };
  inA.addEventListener("input", upd); inB.addEventListener("input", upd);
  const presets = h("div", { class: "row", style: { gap: "8px", marginTop: "8px" } }, ...[[-110, -110], [-150, 130], [-300, 250], [200, -240]].map(([x, y]) => h("button", { class: "btn", onclick: () => { inA.value = x; inB.value = y; upd(); } }, `${x > 0 ? "+" : ""}${x} / ${y > 0 ? "+" : ""}${y}`)));
  el.append(h("div", { class: "row" }, h("label", {}, "Home team ", inA), h("label", {}, "Away team ", inB)), presets, out);

  function draw() {
    const pa = implied(a), pb = implied(b), tot = pa + pb;
    const fa = pa / tot, fb = pb / tot;
    const bar = (v, c, label) => h("div", { style: { width: `${Math.min(100, v * 100 / 1.15)}%`, background: c, color: "#001", padding: "6px 8px", fontWeight: 700, fontSize: ".85rem", whiteSpace: "nowrap", overflow: "hidden" } }, label);
    out.replaceChildren(
      h("div", { style: { display: "flex", height: "34px", borderRadius: "8px", overflow: "hidden", background: "var(--panel2)" } }, bar(pa, "var(--model)", `home ${pct(pa)}`), bar(pb, "var(--blue)", `away ${pct(pb)}`)),
      h("p", { class: "sub", style: { marginTop: "8px" } }, `The two prices add up to ${pct(tot)}. The extra ${pct(tot - 1)} is the bookmaker's margin (the “vig”).`),
      h("div", { class: "found" }, h("b", { class: "k" }, "What the market is really saying"), `Remove the margin proportionally and the market's honest probabilities are ${pct(fa)} home / ${pct(fb)} away. At these prices a bettor with no edge loses about ${pct((tot - 1) / tot)} of every dollar staked on average, which is why beating the closing line is so hard.`));
  }
  draw();
}

const inputStyle = () => ({ width: "100px", padding: "8px 10px", borderRadius: "10px", border: "1px solid var(--line)", background: "var(--panel2)", color: "var(--text)", font: "inherit", fontFamily: "var(--mono)" });
