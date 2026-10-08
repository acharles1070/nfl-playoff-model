import { h, loadJSON, pct } from "../lib.js";

export async function mount(el) {
  const [d, meta] = await Promise.all([loadJSON("ledger"), loadJSON("meta")]);
  const t = d.title, name = (a) => meta.teams[a] || a;
  const rows = t.teams;
  const top = rows[0].p_win_sb;
  el.append(h("h3", {}, `🏆 ${t.season} Super Bowl odds, before week ${t.week}`), h("p", { class: "sub" }, `From 65,536 simulated seasons, logged ${t.logged.slice(0, 10)} (UTC) in a separate hash-chained ledger. I'll score these against the actual champion in February.`));
  el.append(...rows.slice(0, 12).map((r) => h("div", { class: "barrow" },
    h("span", {}, name(r.team)),
    h("div", { class: "track" }, h("div", { style: { width: `${(r.p_win_sb / top) * 100}%`, height: "100%", borderRadius: "6px", background: "var(--model)" } })),
    h("span", { class: "val" }, h("b", {}, pct(r.p_win_sb)), h("span", { style: { color: "var(--muted)" } }, ` title · ${pct(r.p_playoffs, 0)} playoffs`)))));
  el.append(h("details", {}, h("summary", {}, "All 32 teams"), h("div", { class: "scroll" }, h("table", {}, h("thead", {}, h("tr", {}, ...["Team", "Win Super Bowl", "Win conference", "Make playoffs", "Win division", "Expected wins"].map((x, i) => h("th", { class: i ? "num" : "" }, x)))),
    h("tbody", {}, ...rows.map((r) => h("tr", {}, h("td", {}, name(r.team)), ...["p_win_sb", "p_win_conf", "p_playoffs", "p_division"].map((k) => h("td", { class: "num" }, pct(r[k]))), h("td", { class: "num" }, r.expected_wins.toFixed(1)))))))));
  el.append(h("div", { class: "caution", style: { marginTop: "12px" } }, h("b", { class: "k" }, "Heads up"), "Scored on past seasons, my title odds were a little worse than the betting market's (about 0.25 in log score, though with only nine holdout champions that gap could be luck). Early-season title odds are very uncertain by nature, and one season is one champion."));
}
