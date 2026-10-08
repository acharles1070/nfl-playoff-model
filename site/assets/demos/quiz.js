import { h, s, $, loadJSON, plot, responsive, logLoss1, f3, pct, segmented } from "../lib.js";

const ROUNDS = 10;

export async function mount(el) {
  const [games, meta] = await Promise.all([loadJSON("quiz"), loadJSON("meta")]);
  const name = (t) => (meta.teams[t] || t);
  let order, i, guess, locked, you, mod, mkt, hint;

  const reset = () => {
    order = [...games].sort(() => Math.random() - 0.5).slice(0, ROUNDS);
    i = 0; you = []; mod = []; mkt = []; locked = false; guess = 0.55;
    draw();
  };

  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : NaN);
  const fmt = (x) => (Number.isNaN(x) ? "–" : f3(x));

  function board() {
    const row = (label, color, a) => h("div", { class: "stat", style: { borderColor: color, flex: 1, padding: "10px 14px" } },
      h("div", { class: "n", style: { fontSize: "1.5rem", color } }, fmt(mean(a))), h("div", { class: "l" }, label));
    return h("div", { class: "row", style: { gap: "10px", flexWrap: "nowrap" } },
      row("You (avg log loss)", "var(--you)", you), row("My model", "var(--model)", mod), row("Betting market", "var(--market)", mkt));
  }

  function curve(g) {
    const box = h("div", { class: "chart" });
    responsive(box, (w) => {
      const { svg, g: gg, x, y } = plot(w, 190, { xDomain: [0, 1], yDomain: [0, 3], xFmt: (v) => `${Math.round(v * 100)}%`, yFmt: (v) => v.toFixed(1), xLabel: "probability you give to “home team wins”", yLabel: "penalty", margin: { left: 40 } });
      const pts = (fn) => Array.from({ length: 99 }, (_, k) => { const p = (k + 1) / 100; return `${k ? "L" : "M"}${x(p).toFixed(1)},${y(Math.min(3, fn(p))).toFixed(1)}`; }).join("");
      gg.append(s("path", { d: pts((p) => -Math.log(p)), fill: "none", stroke: "var(--blue)", "stroke-width": 2 }), s("path", { d: pts((p) => -Math.log(1 - p)), fill: "none", stroke: "var(--bad)", "stroke-width": 2, "stroke-dasharray": "5 4" }));
      gg.append(s("text", { x: x(0.09), y: y(2.55), style: { fill: "var(--blue)" }, text: "if the home team wins" }), s("text", { x: x(0.98), y: y(2.55), "text-anchor": "end", style: { fill: "var(--bad)" }, text: "if the away team wins" }));
      const mark = (p, color, label, dy) => { const yy = g.home_win ? -Math.log(p) : -Math.log(1 - p); gg.append(s("circle", { cx: x(p), cy: y(Math.min(3, yy)), r: 6, fill: color }), s("text", { x: x(p), y: y(Math.min(3, yy)) - 10 - dy, "text-anchor": "middle", style: { fill: color, fontWeight: 700 }, text: label })); };
      if (locked) { mark(guess, "var(--you)", "you", 0); mark(g.p_model, "var(--model)", "model", 14); mark(g.p_market, "var(--market)", "market", 28); }
      else gg.append(s("line", { x1: x(guess), x2: x(guess), y1: 0, y2: y(0), stroke: "var(--you)", "stroke-dasharray": "3 3" }));
      return svg;
    });
    return box;
  }

  function draw() {
    el.replaceChildren();
    el.append(h("h3", {}, "🎯 Game: beat the model"), h("p", { class: "sub" }, `${ROUNDS} real games from 2017–2025. Set the chance you think the home team won, lock it in, and see how you score against my model and the betting market.`));
    if (i >= ROUNDS) return summary();

    const g = order[i];
    const spread = g.spread == null ? "" : g.spread > 0 ? `${g.home} favored by ${g.spread}` : g.spread < 0 ? `${g.away} favored by ${-g.spread}` : "pick'em";
    el.append(board());
    el.append(h("div", { class: "card", style: { margin: "14px 0" } },
      h("div", { class: "pill" }, `Game ${i + 1} of ${ROUNDS} · ${g.season} ${g.type === "REG" ? `week ${g.week}` : "playoffs"} · ${g.date}`),
      h("div", { style: { fontSize: "1.5rem", fontWeight: 800, margin: "8px 0" } }, `${name(g.away)} @ ${name(g.home)}`),
      h("label", { style: { fontSize: ".88rem", color: "var(--muted)" } }, h("input", { type: "checkbox", checked: hint, onchange: (e) => { hint = e.target.checked; draw(); } }), " Show me the point spread (a hint)"),
      hint ? h("div", { class: "sub", style: { marginTop: "6px" } }, `Spread: ${spread || "n/a"}`) : null));

    const out = h("b", {}, `${Math.round(guess * 100)}%`);
    const input = h("input", { type: "range", min: 1, max: 99, step: 1, value: Math.round(guess * 100), disabled: locked || null });
    input.addEventListener("input", () => { guess = +input.value / 100; out.textContent = `${input.value}%`; if (!locked) redrawCurve(); });
    el.append(h("div", { class: "ctl" }, h("label", {}, h("span", {}, `Chance ${name(g.home)} (home) win`), out), input));

    const holder = h("div", {}); el.append(holder);
    const redrawCurve = () => holder.replaceChildren(curve(g));
    redrawCurve();

    if (!locked) {
      el.append(h("div", { class: "row", style: { marginTop: "12px" } }, h("button", { class: "btn primary", onclick: lock }, "Lock it in")));
    } else {
      const won = g.home_win ? g.home : g.away;
      const L = (p) => logLoss1(p, g.home_win);
      el.append(h("div", { class: "found" }, h("b", { class: "k" }, "Result"),
        `Final: ${g.away} ${g.away_score}, ${g.home} ${g.home_score}. ${name(won)} won.`,
        h("div", { class: "scroll" }, h("table", {}, h("tbody", {},
          row("You", "var(--you)", guess, L(guess)), row("My model", "var(--model)", g.p_model, L(g.p_model)), row("Market", "var(--market)", g.p_market, L(g.p_market)))))));
      el.append(h("div", { class: "row", style: { marginTop: "12px" } }, h("button", { class: "btn primary", onclick: () => { i++; locked = false; guess = 0.55; draw(); } }, i + 1 >= ROUNDS ? "See my score" : "Next game →")));
    }
  }

  const row = (who, color, p, l) => h("tr", {}, h("td", { style: { color, fontWeight: 700 } }, who), h("td", { class: "num" }, `${Math.round(p * 100)}% home`), h("td", { class: "num" }, `penalty ${f3(l)}`));

  function lock() {
    const g = order[i];
    locked = true;
    you.push(logLoss1(guess, g.home_win)); mod.push(logLoss1(g.p_model, g.home_win)); mkt.push(logLoss1(g.p_market, g.home_win));
    draw();
  }

  function summary() {
    el.append(board());
    const verdict = mean(you) < mean(mkt) ? "You beat the betting market this round. Nice (but ten games is a small sample)."
      : mean(you) < mean(mod) ? "You beat my model but not the market."
      : "The market and the model beat you, which is the usual result. Don't feel bad: it takes thousands of games to separate luck from skill.";
    el.append(h("div", { class: "found" }, h("b", { class: "k" }, "Verdict"), verdict,
      h("p", { style: { margin: "8px 0 0", color: "var(--muted)" } }, "For reference: always saying 50% scores 0.693. Over my whole 2,494-game test, the model averages 0.630 and the market 0.607.")));
    el.append(h("button", { class: "btn primary", onclick: reset }, "Play again"));
  }

  hint = false;
  reset();
}
