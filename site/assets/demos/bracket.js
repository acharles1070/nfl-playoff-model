import { h, s, loadJSON, normCdf, mulberry32, gaussian, slider, segmented, pct } from "../lib.js";

const MIN_S2 = 0.09;   // floor on s^2, as in the real simulator (MIN_S = 0.30)

export async function mount(el) {
  const d = await loadJSON("bracket");
  const meta = await loadJSON("meta");
  const T = d.teams, idx = Object.fromEntries(T.map((t, i) => [t, i]));
  const st = { tau: 1, runs: "1", nudgeTeam: T[0], nudge: {} };
  const name = (t) => meta.teams[t] || t;

  // P(home beats away) given this season's strength shocks
  function gameP(home, away, neutral, eps, tauScale) {
    const i = idx[home], j = idx[away];
    const m = (neutral ? d.m_neutral : d.m_home)[i][j] + (st.nudge[home] || 0) - (st.nudge[away] || 0);
    const s2 = Math.max(1 - tauScale ** 2 * (d.tau2[i] + d.tau2[j]), MIN_S2);
    return normCdf((m + eps[i] - eps[j]) / Math.sqrt(s2));
  }

  // The model's own single-game probability (before any season-specific strength shock)
  function marginal(home, away, neutral) {
    const i = idx[home], j = idx[away];
    return normCdf((neutral ? d.m_neutral : d.m_home)[i][j] + (st.nudge[home] || 0) - (st.nudge[away] || 0));
  }

  function playConference(seeds, eps, rng, tau, log) {
    const play = (home, away, round, neutral = false) => {
      const p = gameP(home, away, neutral, eps, tau);
      const homeWins = rng() < p;
      log && log.push({ round, home, away, p: marginal(home, away, neutral), winner: homeWins ? home : away });
      return homeWins ? home : away;
    };
    const rank = Object.fromEntries(seeds.map((t, i) => [t, i + 1]));
    const wc = [[seeds[1], seeds[6]], [seeds[2], seeds[5]], [seeds[3], seeds[4]]].map(([a, b]) => play(a, b, "Wild Card"));
    const alive = [seeds[0], ...wc].sort((a, b) => rank[a] - rank[b]);
    const div = [play(alive[0], alive[3], "Divisional"), play(alive[1], alive[2], "Divisional")].sort((a, b) => rank[a] - rank[b]);
    return play(div[0], div[1], "Conference");
  }

  function season(rng, tau, log) {
    const eps = d.tau2.map((v) => Math.sqrt(v) * tau * gaussian(rng));
    const afc = playConference(d.conferences.AFC, eps, rng, tau, log);
    const nfc = playConference(d.conferences.NFC, eps, rng, tau, log);
    const p = gameP(afc, nfc, true, eps, tau);
    const win = rng() < p;
    log && log.push({ round: "Super Bowl", home: afc, away: nfc, p: marginal(afc, nfc, true), winner: win ? afc : nfc, neutral: true });
    return { afc, nfc, champ: win ? afc : nfc };
  }

  el.append(h("h3", {}, `🏟️ The ${d.season} playoffs, simulated`), h("p", { class: "sub" }, `The real ${d.season} field with the model's pre-playoff game probabilities. Actual champion: ${name(d.champion)}.`));
  const ctl = h("div", { class: "row" });
  const tauCtl = slider({ label: "Uncertainty in team strength", min: 0, max: 1.6, step: 0.1, value: 1, fmt: (v) => `×${v.toFixed(1)}`, onInput: (v) => { st.tau = v; if (st.runs !== "1") run(); } });
  const runSeg = segmented([["1", "Play one playoff"], ["1000", "Run 1,000"], ["10000", "Run 10,000"]], "1", (v) => { st.runs = v; run(); });
  ctl.append(runSeg, tauCtl.el);

  const nudgeRow = h("div", { class: "row", style: { marginTop: "10px" } });
  const sel = h("select", { style: { padding: "8px", borderRadius: "10px", border: "1px solid var(--line)", background: "var(--panel2)", color: "var(--text)", font: "inherit" }, onchange: (e) => { st.nudgeTeam = e.target.value; nudgeCtl.input.value = st.nudge[st.nudgeTeam] || 0; nudgeCtl.out.textContent = (+nudgeCtl.input.value).toFixed(2); } },
    ...T.map((t) => h("option", { value: t }, name(t))));
  const nudgeCtl = slider({ label: "What if that team were stronger/weaker? (probit units)", min: -1, max: 1, step: 0.05, value: 0, fmt: (v) => (v > 0 ? "+" : "") + v.toFixed(2), onInput: (v) => { st.nudge[st.nudgeTeam] = v; if (st.runs !== "1") run(); } });
  nudgeRow.append(h("label", {}, "Team ", sel), nudgeCtl.el, h("button", { class: "btn", onclick: () => { st.nudge = {}; nudgeCtl.input.value = 0; nudgeCtl.out.textContent = "0.00"; run(); } }, "Reset"));

  const out = h("div", { style: { marginTop: "14px" } });
  el.append(ctl, nudgeRow, out);

  function run() {
    const n = +st.runs;
    if (n === 1) return one();
    const rng = mulberry32(2025);
    const champ = {}, conf = {};
    for (let k = 0; k < n; k++) {
      const r = season(rng, st.tau);
      champ[r.champ] = (champ[r.champ] || 0) + 1;
      conf[r.afc] = (conf[r.afc] || 0) + 1; conf[r.nfc] = (conf[r.nfc] || 0) + 1;
    }
    const rows = T.map((t) => ({ t, c: (champ[t] || 0) / n, f: (conf[t] || 0) / n })).sort((a, b) => b.c - a.c);
    const top = rows[0].c;
    out.replaceChildren(
      h("div", { class: "sub", style: { marginBottom: "4px" } }, "Chance to win the Super Bowl (and, in grey, to reach it):"),
      h("div", {}, ...rows.slice(0, 10).map((r) => h("div", { class: "barrow" },
        h("span", { style: { fontWeight: r.t === d.champion ? 800 : 500 } }, name(r.t), r.t === d.champion ? " 🏆" : ""),
        h("div", { class: "track" }, h("div", { style: { width: `${(r.c / top) * 100}%`, height: "100%", borderRadius: "6px", background: r.t === d.champion ? "var(--market)" : "var(--model)" } })),
        h("span", { class: "val" }, h("b", {}, pct(r.c)), h("span", { style: { color: "var(--muted)" } }, `  reach ${pct(r.f, 0)}`))))),
      h("div", { class: "sub", style: { marginTop: "8px" } }, `${n.toLocaleString()} simulated playoffs. ${name(d.champion)} (the real champion 🏆) had a ${pct((champ[d.champion] || 0) / n)} chance in this simulation. The favorite was ${name(rows[0].t)} at ${pct(rows[0].c)}: even the best team is far from a lock. (Sampling noise is about ±${(Math.sqrt(0.15 * 0.85 / n) * 100 * 1.96).toFixed(1)} points; my full simulator solves the bracket exactly instead of sampling.)`));
  }

  function one() {
    const log = [];
    const r = season(mulberry32((Math.random() * 2 ** 32) >>> 0), st.tau, log);
    const rounds = ["Wild Card", "Divisional", "Conference", "Super Bowl"];
    out.replaceChildren(h("div", { class: "grid2" }, ...rounds.map((round) => h("div", { class: "card" }, h("b", {}, round),
      ...log.filter((g) => g.round === round).map((g) => h("div", { style: { margin: "6px 0", fontSize: ".9rem" } },
        h("span", { style: { fontWeight: g.winner === g.home ? 800 : 400 } }, g.home), " vs ", h("span", { style: { fontWeight: g.winner === g.away ? 800 : 400 } }, g.away),
        h("span", { class: "pill", style: { marginLeft: "6px" }, title: "the model's odds before this simulated season's strength shocks" }, g.neutral ? `${pct(g.p, 0)} (AFC side)` : `${pct(g.p, 0)} home`), " → ", h("b", { style: { color: "var(--model)" } }, g.winner)))))),
      h("div", { class: "found", style: { marginTop: "10px" } }, h("b", { class: "k" }, "This simulated season"), `${name(r.champ)} win the Super Bowl. The percentages are the model\'s odds for each game going in; in each simulated season every team is also nudged up or down by a random amount (the uncertainty slider), which is why favorites sometimes lose. Press the button again for a different universe, or run thousands to see the odds.`));
  }
  run();
}
