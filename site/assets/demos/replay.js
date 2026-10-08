import { h, s, loadJSON, plot, responsive, slider, segmented, pathFrom, pct } from "../lib.js";

const ROUNDS = ["WC", "DIV", "CON", "SB"];
const RNAME = { WC: "Wild Card", DIV: "Divisional", CON: "Conference", SB: "Super Bowl" };
const ord = (n) => `${n}${[11, 12, 13].includes(n % 100) ? "th" : ["th", "st", "nd", "rd"][n % 10 > 3 ? 0 : n % 10]}`;
const ok = (v) => h("b", { style: { color: v ? "var(--accent)" : "var(--bad)" } }, v ? "✓" : "✗");

// 95% Wilson interval for k successes in n trials
function wilson(k, n) {
  const z = 1.96, p = k / n, d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d, m = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return [c - m, c + m];
}

export async function mount(el) {
  const [r, meta] = await Promise.all([loadJSON("bracket_replay"), loadJSON("meta")]);
  const name = (t) => meta.teams[t] || t;
  const W = r.weekly, S = r.summary, B = r.bracket_scores;
  const champ = r.champion;
  const actualSeed = {};
  for (const conf of ["AFC", "NFC"]) r.actual_field[conf].forEach((x) => (actualSeed[x.team] = x.seed));
  let idx = 8, mode = "model", timer = null;

  // ================= part 1: the time machine ======================================================
  el.append(h("h3", {}, "⏪ Rewind the 2025 season"), h("p", { class: "sub" }, "The playoffs in January 2026 belonged to the 2025 season. Slide to any week: the left column is the bracket field I would have projected using only games before that week, the right column is what really happened."));
  const readout = h("div", { class: "grid2", style: { marginBottom: "10px" } });
  const tables = h("div", { class: "grid2" });
  const chart = h("div", { class: "chart" });
  const ctl = slider({ label: "Rewind to", min: 0, max: W.length - 1, step: 1, value: idx, fmt: (v) => W[v].label, onInput: (v) => { idx = v; draw(); } });
  const seg = segmented([["model", "My model's projection"], ["standings", "Naive baseline: the standings"]], mode, (v) => { mode = v; draw(); });
  const play = h("button", { class: "btn", onclick: () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ Play the season"; return; }
    play.textContent = "⏸ Pause";
    idx = 0;
    timer = setInterval(() => { idx = idx >= W.length - 1 ? 0 : idx + 1; ctl.input.value = idx; ctl.out.textContent = W[idx].label; draw(); }, 900);
  } }, "▶ Play the season");
  el.append(h("div", { class: "row" }, ctl.el, h("div", {}, seg), play), readout, tables, chart,
    h("div", { class: "legend" }, h("span", {}, h("i", { style: { background: "var(--model)" } }), "my model: playoff teams correctly in the field"), h("span", {}, h("i", { style: { background: "var(--blue)" } }), "standings baseline (ties broken by point differential, not the NFL's full rules, so it can miss one at the very end)")));

  function draw() {
    const w = W[idx];
    const proj = mode === "model" ? w.field : w.baseline_field;
    const sc = mode === "model" ? w.model : w.standings;
    const sea = (w.title_top.find((x) => x.team === champ) || {}).p;

    readout.replaceChildren(
      h("div", { class: "card" }, h("b", {}, `${w.label}: `), `the ${mode === "model" ? "model" : "standings baseline"} had `, h("b", {}, `${sc.in_field} of the 14`), " eventual playoff teams in the field and ", h("b", {}, `${sc.exact_seed}`), " in exactly the right seed. ",
        h("span", { style: { color: "var(--muted)" } }, `(${mode === "model" ? "standings baseline" : "model"}: ${mode === "model" ? w.standings.in_field : w.model.in_field} and ${mode === "model" ? w.standings.exact_seed : w.model.exact_seed}.)`)),
      h("div", { class: "card" }, `The model's title favorite was `, h("b", {}, name(w.title_top[0].team)), ` at ${pct(w.title_top[0].p, 1)}. The eventual champion, ${name(champ)}, was ${sea !== undefined ? `at ${pct(sea, 1)}` : "outside its top six"}. Its qualification forecasts scored ${w.qual_log_loss.toFixed(3)} log loss (always saying the base rate scores ${r.qualification_context ? r.qualification_context.base_rate_entropy.toFixed(3) : "0.673"}).`));

    tables.replaceChildren(...["AFC", "NFC"].map((conf) => h("div", { class: "card" }, h("b", {}, `${conf}: projected vs actual`),
      h("table", { style: { marginTop: "6px" } }, h("thead", {}, h("tr", {}, ...["Seed", mode === "model" ? "Model" : "Standings", "Actual", ""].map((t) => h("th", {}, t)))),
        h("tbody", {}, ...proj[conf].map((team, k) => {
          const real = r.actual_field[conf][k].team;
          const exact = team === real, inField = actualSeed[team] !== undefined;
          return h("tr", {}, h("td", {}, k + 1), h("td", {}, team), h("td", {}, real),
            h("td", { title: exact ? "right team, right seed" : inField ? `made the playoffs, as the ${actualSeed[team]} seed` : "missed the playoffs" },
              exact ? ok(true) : inField ? h("b", { style: { color: "var(--warn)" } }, `↕ ${actualSeed[team]}`) : ok(false)));
        }))))));

    responsive(chart, (width) => {
      const { svg, g, x, y } = plot(width, 210, { xDomain: [1, W.length], yDomain: [4, 14], xTicks: W.map((_, i) => i + 1).filter((v) => v % 3 === 1 || v === W.length), xFmt: (v) => (v === W.length ? "end" : v), yFmt: (v) => v, xLabel: "week (projection made before that week)", yLabel: "of 14 in the field", margin: { left: 44 } });
      const line = (key, color) => g.append(s("path", { d: pathFrom(W.map((q, i) => [x(i + 1), y(q[key].in_field)])), fill: "none", stroke: color, "stroke-width": 2.6 }));
      line("standings", "var(--blue)"); line("model", "var(--model)");
      g.append(s("line", { x1: x(idx + 1), x2: x(idx + 1), y1: 0, y2: y(4), stroke: "var(--you)", "stroke-dasharray": "4 3" }),
        s("circle", { cx: x(idx + 1), cy: y(W[idx].model.in_field), r: 5.5, fill: "var(--model)" }), s("circle", { cx: x(idx + 1), cy: y(W[idx].standings.in_field), r: 5.5, fill: "var(--blue)" }));
      return svg;
    });
  }
  draw();

  // ================= part 2: the bracket before kickoff ===============================================
  const winners = (b) => Object.fromEntries(ROUNDS.map((rd) => [rd, rd === "SB" ? [b.SB.winner] : ["AFC", "NFC"].flatMap((c) => b[c][rd].map((g) => g.winner))]));
  const pick = winners(r.chalk_bracket), real = winners(r.actual_bracket);
  const upsets = ["AFC", "NFC"].flatMap((c) => ["WC", "DIV", "CON"].flatMap((rd) => r.actual_bracket[c][rd])).concat([r.actual_bracket.SB]).filter((g) => g.upset).length;

  el.append(h("h3", { style: { marginTop: "26px" } }, "📋 The whole bracket, picked before kickoff"),
    h("p", { class: "sub" }, "Before the Wild Card round I fill in every game with the model's favorite (reseeding as the NFL does), without knowing any result. A pick counts if that team wins in that round, even if the matchup turned out differently. Baseline: the higher seed always wins."),
    h("div", { class: "scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ...["Round", "Model's picks", "Who actually won", "Model", "Higher seed"].map((t, i) => h("th", { class: i > 2 ? "num" : "" }, t)))),
      h("tbody", {}, ...ROUNDS.map((rd) => h("tr", {}, h("td", {}, RNAME[rd]),
        h("td", {}, ...pick[rd].map((t, i) => h("span", { style: { marginRight: "8px", color: real[rd].includes(t) ? "var(--accent)" : "var(--muted)", fontWeight: real[rd].includes(t) ? 700 : 400 } }, t))),
        h("td", {}, real[rd].join("  ")), h("td", { class: "num" }, `${B.model[rd]}/${B.games[rd]}`), h("td", { class: "num" }, `${B.higher_seed[rd]}/${B.games[rd]}`))),
        h("tr", { style: { fontWeight: 800 } }, h("td", {}, "Total"), h("td", {}, `champion pick: ${name(r.chalk_bracket.champion)}`), h("td", {}, `champion: ${name(champ)}`), h("td", { class: "num" }, `${B.model.total}/13`), h("td", { class: "num" }, `${B.higher_seed.total}/13`))))));

  // ================= part 3: round by round, real matchups ===============================================
  const best = r.preplayoff_title_odds.slice().sort((a, b) => b.p_win_sb - a.p_win_sb);
  const sbRank = best.findIndex((x) => x.team === champ) + 1;
  el.append(h("h3", { style: { marginTop: "26px" } }, "🎯 Game by game, with the real matchups"),
    h("p", { class: "sub" }, "Each game as it was set: my model's probability, the closing betting line, and the result. Picks are whichever side is above 50%."),
    h("div", { class: "scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ...["Round", "Home", "Away", "Score", "My model: home win", "Closing line", "Model", "Line", "Higher seed"].map((t, i) => h("th", { class: i >= 4 ? "num" : "" }, t)))),
      h("tbody", {}, ...r.games.map((g) => h("tr", {}, h("td", {}, RNAME[g.round]), h("td", { style: { fontWeight: g.home_won ? 800 : 400 } }, g.home), h("td", { style: { fontWeight: g.home_won ? 400 : 800 } }, g.away),
        h("td", { class: "num" }, `${g.home_score}–${g.away_score}`), h("td", { class: "num" }, pct(g.p_model)), h("td", { class: "num" }, g.p_market === null ? "–" : pct(g.p_market)),
        h("td", { class: "num" }, ok(g.model_right)), h("td", { class: "num" }, g.market_right === null ? "–" : ok(g.market_right)), h("td", { class: "num" }, ok(g.seed_right))))))));

  // ================= part 4: the scorecard ==================================================================
  const [lo, hi] = wilson(S.model_right, S.n_games);
  const ahead = W.filter((w) => w.model.in_field > w.standings.in_field).length, behind = W.filter((w) => w.model.in_field < w.standings.in_field).length;
  const qc = r.qualification_context;
  const first = W[0], last = W[W.length - 1];
  el.append(h("h3", { style: { marginTop: "26px" } }, "🧾 How it did"),
    h("div", { class: "scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ...["The 13 playoff games", "Winners picked right", "Log loss (lower is better)"].map((t, i) => h("th", { class: i ? "num" : "" }, t)))),
      h("tbody", {},
        h("tr", {}, h("td", {}, "My model"), h("td", { class: "num" }, `${S.model_right}/${S.n_games}`), h("td", { class: "num" }, S.log_loss_model.toFixed(3))),
        h("tr", {}, h("td", {}, "Closing betting line"), h("td", { class: "num" }, `${S.market_right}/${S.market_games}`), h("td", { class: "num" }, S.log_loss_market === null ? "–" : S.log_loss_market.toFixed(3))),
        h("tr", {}, h("td", {}, "Higher seed always wins"), h("td", { class: "num" }, `${S.seed_right}/${S.n_games}`), h("td", { class: "num" }, "–")),
        h("tr", {}, h("td", {}, "Coin flip"), h("td", { class: "num" }, "6.5/13"), h("td", { class: "num" }, S.log_loss_coin.toFixed(3)))))),
    h("div", { class: "found", style: { marginTop: "14px" } }, h("b", { class: "k" }, "What I found"),
      h("ul", { style: { margin: "6px 0 0", paddingLeft: "20px" } },
        h("li", {}, h("b", {}, "The field. "), `Before week 1 my model had ${first.model.in_field} of the 14 eventual playoff teams in its projected field; the standings baseline (last year's field) had ${first.standings.in_field}. By the last projection the model had ${last.model.in_field} and the standings baseline ${last.standings.in_field}. Across the ${W.length} projections the model was ahead of the standings in ${ahead}, behind in ${behind} and level in ${W.length - ahead - behind}. So the simulation did not clearly beat "just look at the standings" this year.`),
        qc ? h("li", {}, h("b", {}, "Context. "), (() => {
          const c = (wk) => qc.checkpoints.find((x) => x.week === wk), w1 = c(1), w9 = c(9);
          const beatsAll = qc.checkpoints.every((x) => x.all_seasons < qc.base_rate_entropy);
          return `2025 was a hard year to project. My qualification forecasts scored ${w1.this_season.toFixed(2)} log loss at week 1 and ${w9.this_season.toFixed(2)} at week 9, against ${w1.all_seasons.toFixed(2)} and ${w9.all_seasons.toFixed(2)} averaged over ${qc.n_seasons} seasons (${qc.seasons}). Always guessing the league-wide base rate scores ${qc.base_rate_entropy.toFixed(2)}.${beatsAll ? " Across all those seasons the model beats that no-skill guess at every checkpoint" : ""}${w1.this_season > qc.base_rate_entropy ? "; in 2025 it did not before week 1." : "."}`;
        })()) : null,
        h("li", {}, h("b", {}, "The bracket. "), `Picked before kickoff, the model's bracket got ${B.model.total} of 13 winners right, the higher-seed bracket ${B.higher_seed.total}. It picked ${name(r.chalk_bracket.champion)} to win it all; ${name(champ)} did. Before the playoffs it gave ${name(champ)} ${pct(S.champion_title_odds, 1)}, ${sbRank === 1 ? "the best odds in the field" : `${ord(sbRank)} best of 14`}.`),
        h("li", {}, h("b", {}, "The games. "), `With the real matchups the model picked ${S.model_right} of ${S.n_games} winners, the closing line ${S.market_right}, the higher seed ${S.seed_right}. There were ${upsets} upsets by the model's reckoning. The market's log loss was a little better (${S.log_loss_market}, against ${S.log_loss_model}).`))),
    h("div", { class: "caution" }, h("b", { class: "k" }, "Thirteen games is nothing"),
      `A ${S.model_right}-of-${S.n_games} record has a 95% interval of roughly ${(lo * 100).toFixed(0)}% to ${(hi * 100).toFixed(0)}% accuracy, and one game separates the model from the market. This replay shows what one bracket looked like, not whether the model is skilled. The honest evidence about skill is the season-long scorecard in Chapter 7.`),
    h("div", { class: "plain" }, h("b", { class: "k" }, "See it live"), "The same machinery now runs on the 2026 season, refreshing every morning: ", h("a", { href: "bracket/" }, "the live bracket for the January 2027 playoffs →")));
}
