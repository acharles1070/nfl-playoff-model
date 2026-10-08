// Draws the live bracket from data/bracket.json (regenerated daily by a GitHub Actions job).
const $ = (s, el = document) => el.querySelector(s);
const h = (tag, attrs = {}, ...kids) => {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k === "style") el.setAttribute("style", v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat(Infinity)) if (kid !== null && kid !== undefined && kid !== false) el.append(kid.nodeType ? kid : document.createTextNode(String(kid)));
  return el;
};

const pct = (x, d = 0) => (x === 0 ? "–" : x < 0.005 && d === 0 ? "<1%" : `${(x * 100).toFixed(d)}%`);
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const ROUND = { WC: "Wild Card", DIV: "Divisional", CON: "Conference", SB: "Super Bowl" };
let D, byTeam = {};

const when = (k) => {
  if (!k) return "";
  const [d, t] = k.split(" ");
  const dt = new Date(`${d}T12:00:00`);
  const [hh, mm] = (t || "").split(":").map(Number);
  const clock = t ? ` · ${((hh + 11) % 12) + 1}:${String(mm).padStart(2, "0")} ${hh >= 12 ? "PM" : "AM"} ET` : "";
  return `${DAYS[dt.getDay()]} ${MONTHS[dt.getMonth()]} ${dt.getDate()}${clock}`;
};
const stamp = (iso) => new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });

// ---- hero ---------------------------------------------------------------------------------
function hero() {
  const alive = D.teams.filter((t) => t.in_field !== false && !t.eliminated);
  const top = [...alive].sort((a, b) => b.p_win_sb - a.p_win_sb);
  $("#eyebrow").textContent = `${D.season} season · playoffs in January ${D.playoffs_year}`;
  const champ = byTeam[D.bracket.champion];
  $("#title").textContent = D.phase === "complete" ? `${champ.name} are the champions` : D.phase === "playoffs" ? "Playoff bracket" : "Projected playoff bracket";

  const through = D.data_through ? `Data through week ${D.data_through.week} (${D.data_through.date})` : "";
  $("#status").replaceChildren(h("span", { class: `pill ${D.phase === "regular_season" ? "warn" : "live"}` }, D.label), through && h("span", { class: "pill" }, through), h("span", { class: "pill" }, `Updated ${stamp(D.generated_utc)}`));
  $("#lede").textContent = D.phase === "regular_season"
    ? "No playoff schedule exists yet, so this is a projection: the bracket the model expects if the season plays out the way its simulations say is most likely. It refreshes every morning and the matchups will change."
    : D.phase === "playoffs" ? "The field is real. Finished games show their scores; the rest show the model's win probability, and title odds are re-simulated from the teams still alive."
    : "The season is complete. Every game shows the model's pregame probability next to the result.";

  const cards = [];
  if (D.phase !== "complete") cards.push(["Most likely champion", top[0].name, `${pct(top[0].p_win_sb, 1)} to win the Super Bowl`]);
  else cards.push(["Champion", champ.name, `${D.bracket.SB.home_score ?? ""}–${D.bracket.SB.away_score ?? ""} in the Super Bowl`]);
  if (D.phase === "regular_season") {
    for (const conf of ["AFC", "NFC"]) {
      const f = D.field[conf][0];          // the bracket's own 1 seed, so the page agrees with itself
      cards.push([`${conf} projected 1 seed`, f.name, `${pct(f.p_seed)} chance of the first-round bye`]);
    }
    const mover = [...D.teams].filter((t) => t.d_win_sb !== undefined).sort((a, b) => Math.abs(b.d_win_sb) - Math.abs(a.d_win_sb))[0];
    if (mover) cards.push(["Biggest mover this week", mover.name, `title odds ${mover.d_win_sb >= 0 ? "up" : "down"} ${Math.abs(mover.d_win_sb * 100).toFixed(1)} points to ${pct(mover.p_win_sb, 1)}`]);
  } else {
    const games = [...D.bracket.AFC.WC, ...D.bracket.AFC.DIV, ...D.bracket.AFC.CON, ...D.bracket.NFC.WC, ...D.bracket.NFC.DIV, ...D.bracket.NFC.CON, D.bracket.SB];
    const done = games.filter((g) => g.status === "final");
    cards.push(["Games played", `${done.length} of 13`, `${done.filter((g) => g.upset).length} upsets (the model's underdog won)`]);
    if (D.phase === "playoffs" && top[1]) cards.push(["Next", D.label.replace("Playoffs: ", ""), `${top[0].name} ${pct(top[0].p_win_sb, 0)}, ${top[1].name} ${pct(top[1].p_win_sb, 0)} to win it all`]);
  }
  cards.push(["Bracket's champion pick", champ.name, D.phase === "regular_season" ? "favorite in every game (not the same as the likeliest champion)" : "the favorite in every remaining game"]);
  $("#cards").replaceChildren(...cards.map(([k, v, s]) => h("div", { class: "card" }, h("div", { class: "k" }, k), h("div", { class: "v" }, v), h("div", { class: "s" }, s))));
}

// ---- bracket ------------------------------------------------------------------------------
function teamRow(g, side) {
  const team = side === "home" ? g.home : g.away;
  const p = side === "home" ? g.p_home : 1 - g.p_home;
  const seed = side === "home" ? g.home_seed : g.away_seed;
  const decided = g.status === "final";
  const cls = decided ? (g.winner === team ? "win" : "lose") : g.favorite === team ? "pick" : "";
  const score = side === "home" ? g.home_score : g.away_score;
  return h("div", { class: `team ${cls}`, title: `${byTeam[team].name}: ${pct(p, 1)} to win` },
    h("span", { class: "bar", style: `--p:${(p * 100).toFixed(0)}%` }), h("span", { class: "seed" }, seed ?? ""),
    h("span", { class: "nm" }, team, h("small", {}, byTeam[team].record)), h("span", { class: "pr" }, decided ? `${score}` : pct(p)));
}

function gameCard(g) {
  const right = g.status === "final" ? "Final" : when(g.kickoff) || (g.status === "scheduled" ? "Scheduled" : "Projected");
  const left = g.status === "final" ? `model had ${g.favorite} at ${pct(Math.max(g.p_home, 1 - g.p_home))}` : g.round === "SB" ? "neutral site" : `${g.home} hosts`;
  return h("div", { class: `game ${g.status}` },
    h("div", { class: "meta" }, h("span", {}, ROUND[g.round]), h("span", {}, right)),
    teamRow(g, "home"), teamRow(g, "away"),
    h("div", { class: "foot" }, h("span", {}, left), g.upset ? h("span", { class: "upset" }, "⚡ upset") : ""));
}

function bracket() {
  const note = D.phase === "regular_season"
    ? "These are not scheduled games. The seeds are the likeliest lineup from the season simulation, and each game shows the model's probability for that hypothetical matchup using today's ratings."
    : D.phase === "playoffs" ? "Solid borders are real games. Dashed ones depend on who advances, so they are the favorite's path." : "";
  $("#bracket-note").textContent = note;
  const conf = (name, cls) => h("div", { class: `conf ${cls}` },
    ...["WC", "DIV", "CON"].map((r) => h("div", { class: `round ${r.toLowerCase()}` }, h("h4", {}, h("span", {}, name), ` ${ROUND[r]}`), ...D.bracket[name][r].map(gameCard))));
  const sb = D.bracket.SB;
  const center = h("div", { class: "center" }, h("h4", {}, "Super Bowl"), gameCard(sb),
    h("div", { class: "trophy" }, D.phase === "complete" ? "Champion" : "Bracket's pick", h("b", {}, `🏆 ${byTeam[D.bracket.champion].name}`)));
  $("#bracket").replaceChildren(conf("AFC", "afc"), center, conf("NFC", "nfc"));
}

// ---- seed odds -------------------------------------------------------------------------------
function seedTables() {
  const wrap = $("#seed-tables");
  wrap.replaceChildren();
  for (const conf of ["AFC", "NFC"]) {
    const field = D.field[conf].map((x) => x.team);
    const rows = D.teams.filter((t) => t.conf === conf && t.p_playoffs > 0.005).sort((a, b) => b.p_playoffs - a.p_playoffs || a.team.localeCompare(b.team));
    const head = h("tr", {}, h("th", {}, "Team"), ...[1, 2, 3, 4, 5, 6, 7].map((k) => h("th", {}, k)), h("th", {}, "Out"));
    const body = rows.map((t) => {
      const out = Math.max(0, 1 - t.p_seed.reduce((a, b) => a + b, 0));
      const cell = (p, mine) => h("td", { class: "heat", style: `background:color-mix(in srgb, var(--accent) ${Math.min(85, p * 100 * 1.1).toFixed(0)}%, transparent);${mine ? "outline:2px solid var(--accent);outline-offset:-2px;font-weight:800;" : ""}` }, p < 0.005 ? "" : pct(p));
      return h("tr", {}, h("td", {}, h("b", {}, t.team), ` ${t.record}`), ...t.p_seed.map((p, k) => cell(p, field[k] === t.team)), h("td", { class: "heat", style: `color:var(--muted)` }, out < 0.005 ? "" : pct(out)));
    });
    wrap.append(h("div", { class: "tcard" }, h("h3", {}, h("span", { style: `color:var(--${conf.toLowerCase()})` }, conf), " seed odds"), h("div", { class: "scroll" }, h("table", {}, h("thead", {}, head), h("tbody", {}, body)))));
  }
}

// ---- all teams ---------------------------------------------------------------------------------
const COLS = [
  ["Team", (t) => t.name, (t) => t.name, true],
  ["Record", (t) => t.record, (t) => t.wins, false],
  ["Exp. wins", (t) => t.expected_wins.toFixed(1), (t) => t.expected_wins, false],
  ["Playoffs", (t) => pct(t.p_playoffs), (t) => t.p_playoffs, false],
  ["Division", (t) => pct(t.p_division), (t) => t.p_division, false],
  ["Bye", (t) => pct(t.p_first_seed), (t) => t.p_first_seed, false],
  ["Win conf.", (t) => pct(t.p_win_conf), (t) => t.p_win_conf, false],
  ["Title", (t) => h("span", {}, h("span", { class: "bar", style: `width:${Math.max(1, t.p_win_sb * 260)}px` }), pct(t.p_win_sb, 1)), (t) => t.p_win_sb, false],
  ["Change", (t) => (t.d_win_sb === undefined ? "" : h("span", { class: t.d_win_sb >= 0 ? "pos" : "neg" }, `${t.d_win_sb >= 0 ? "+" : "−"}${Math.abs(t.d_win_sb * 100).toFixed(1)}`)), (t) => t.d_win_sb ?? 0, false],
];
let sortCol = 7, sortDir = -1;
function teamsTable() {
  const rows = [...D.teams].sort((a, b) => {
    const [, , key, isText] = COLS[sortCol];
    return sortDir * (isText ? key(a).localeCompare(key(b)) : key(a) - key(b));
  });
  const hasChange = D.teams.some((t) => t.d_win_sb !== undefined);
  const cols = COLS.filter((c) => c[0] !== "Change" || hasChange);
  $("#teams-table").replaceChildren(
    h("thead", {}, h("tr", {}, ...cols.map((c, i) => h("th", { class: COLS.indexOf(c) === sortCol ? "sorted" : "", onclick: () => { const j = COLS.indexOf(c); sortDir = j === sortCol ? -sortDir : (c[3] ? 1 : -1); sortCol = j; teamsTable(); } }, c[0] + (COLS.indexOf(c) === sortCol ? (sortDir < 0 ? " ▼" : " ▲") : ""))))),
    h("tbody", {}, ...rows.map((t) => h("tr", {}, ...cols.map((c) => h("td", {}, c[1](t)))))));
}

// ---- chrome -----------------------------------------------------------------------------------
function chrome() {
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll("#tabs button").forEach((x) => x.classList.toggle("on", x === b));
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("on", v.id === `view-${b.dataset.v}`));
  }));
  $("#theme").addEventListener("click", () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* storage may be unavailable */ }
  });
  $("#foot").textContent = `Game model ${D.model.game_model} · season simulator ${D.model.season_simulator} · ${D.n_sims.toLocaleString()} simulated seasons · generated ${stamp(D.generated_utc)}`;
}

fetch("data/bracket.json", { cache: "no-cache" })
  .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
  .then((d) => { D = d; D.teams.forEach((t) => (byTeam[t.team] = t)); hero(); bracket(); seedTables(); teamsTable(); chrome(); })
  .catch((e) => { $("#title").textContent = "Could not load the bracket"; $("#lede").textContent = `The data file did not load (${e.message}). Try again in a few minutes.`; });
