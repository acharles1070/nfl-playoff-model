import { h, slider } from "../lib.js";

const FIRST = 1999, LAST = 2025;

export async function mount(el) {
  let season = 2019, timer = null;
  el.append(h("h3", {}, "⏩ Walk forward through time"), h("p", { class: "sub" }, "Pick a season to predict. Green seasons are the only ones the model may learn from; everything later is hidden."));
  const strip = h("div", { style: { display: "flex", gap: "3px", margin: "14px 0 6px" } });
  const seg = (n, color, label) => h("div", { style: { flex: n, borderTop: `3px solid ${color}`, paddingTop: "3px", textAlign: "center", fontSize: ".74rem", color: "var(--muted)", whiteSpace: "nowrap", overflow: "hidden" } }, label);
  const bands = h("div", { style: { display: "flex", gap: "3px" } }, seg(3, "transparent", ""), seg(15, "var(--blue)", "development window 2002–2016"), seg(9, "var(--warn)", "holdout 2017–2025"));
  const readout = h("div", { class: "found" });

  const years = Array.from({ length: LAST - FIRST + 1 }, (_, k) => FIRST + k);
  const blocks = years.map((y) => h("div", { title: String(y), style: { flex: 1, height: "44px", borderRadius: "5px", display: "flex", alignItems: "flex-end", justifyContent: "center", fontSize: ".62rem", paddingBottom: "2px", color: "#001", transition: "background .25s" } }));
  blocks.forEach((b) => strip.append(b));

  const ctl = slider({ label: "Season to predict", min: 2002, max: LAST, step: 1, value: season, onInput: (v) => { season = v; paint(); } });
  const play = h("button", { class: "btn", onclick: () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ Play"; return; }
    play.textContent = "⏸ Pause";
    timer = setInterval(() => { season = season >= LAST ? 2002 : season + 1; ctl.input.value = season; ctl.out.textContent = season; paint(); }, 650);
  } }, "▶ Play");

  function paint() {
    years.forEach((y, k) => {
      const b = blocks[k];
      b.style.background = y < season ? "var(--model)" : y === season ? "var(--blue)" : "var(--panel2)";
      b.style.opacity = y > season ? 0.55 : 1;
      b.textContent = (y - FIRST) % 4 === 0 || y === season ? String(y).slice(2) : "";
    });
    const train = season - FIRST;
    const where = season >= 2017 ? "holdout (used to score and confirm; every idea was also scored here, so results are corrected for how many I tried)" : "development window (where ideas are selected, since the split was adopted on Oct 6)";
    readout.replaceChildren(h("b", { class: "k" }, `Predicting ${season}`), `The model has learned from ${train} earlier season${train === 1 ? "" : "s"} (${FIRST}–${season - 1}) and knows nothing about ${season} or later. This season is in the ${where}.`);
  }

  el.append(strip, bands, h("div", { class: "row", style: { marginTop: "12px" } }, ctl.el, play), readout,
    h("div", { class: "legend" }, ...[["var(--model)", "learned from"], ["var(--blue)", "being predicted"], ["var(--panel2)", "hidden (the future)"]].map(([c, t]) => h("span", {}, h("i", { style: { background: c, border: "1px solid var(--line)" } }), t))));
  paint();
}
