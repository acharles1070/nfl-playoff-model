import { $, $$, h, loadJSON, f4, pct } from "./lib.js";
import { GLOSSARY } from "./glossary.js";

const REPO_URL = "https://github.com/acharles1070/nfl-playoff-model";

// ---- theme
const root = document.documentElement;
$("#theme").addEventListener("click", () => {
  const dark = root.dataset.theme ? root.dataset.theme === "dark" : !matchMedia("(prefers-color-scheme: light)").matches;
  root.dataset.theme = dark ? "light" : "dark";
  try { localStorage.setItem("theme", root.dataset.theme); } catch (e) { /* storage may be unavailable */ }
  window.dispatchEvent(new Event("themechange"));
});

// ---- nav, progress, scroll-spy
const chapters = $$("section.chapter");
const nav = $("#nav");
chapters.forEach((c, i) => nav.append(h("a", { href: `#${c.id}`, "data-id": c.id }, `${i + 1}. ${c.dataset.title}`)));
const links = $$("a", nav);

function onScroll() {
  const max = document.documentElement.scrollHeight - innerHeight;
  $("#progress").style.width = `${Math.min(100, (scrollY / Math.max(1, max)) * 100)}%`;
  let current = null;
  for (const c of chapters) if (c.getBoundingClientRect().top < innerHeight * 0.4) current = c.id;
  links.forEach((a) => {
    const on = a.dataset.id === current;
    a.classList.toggle("on", on);
    if (on && nav.scrollWidth > nav.clientWidth) nav.scrollTo({ left: a.offsetLeft - 60, behavior: "auto" });
  });
}
addEventListener("scroll", onScroll, { passive: true });
onScroll();

// ---- reveal on scroll
const io = new IntersectionObserver((entries) => entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); } }), { threshold: 0.05 });
$$(".reveal").forEach((el) => io.observe(el));

// ---- glossary tooltips
$$(".term[data-t]").forEach((el) => {
  const g = GLOSSARY[el.dataset.t];
  if (g) { el.dataset.def = g[1]; el.tabIndex = 0; }
});
$("#glossary").append(...Object.entries(GLOSSARY).map(([k, [name, def]]) => h("div", { class: "card", id: `g-${k}` }, h("b", {}, name), h("p", { style: { margin: "4px 0 0", color: "var(--muted)", fontSize: ".92rem" } }, def))));

// ---- repo links
$$("a.repo").forEach((a) => {
  if (REPO_URL) a.href = `${REPO_URL}/${a.dataset.path || ""}`;
  else { a.title = "Set REPO_URL in site/assets/app.js"; }
});

// ---- hero stats, lessons, footer (all from the data files)
const [score, ledger, reg, bets, meta] = await Promise.all(["scorecard", "ledger", "registry", "betting", "meta"].map(loadJSON));
const champ = score.models.find((m) => m.key === "p_champion");
const mkt = score.models.find((m) => m.key === "p_market");
const home = score.models.find((m) => m.key === "p_home_only");
const best = bets.series["0.0"];

$("#hero-stats").append(...[
  [f4(champ.log_loss), `my best model's score (log loss, lower is better). The betting market scores ${f4(mkt.log_loss)}`],
  [score.games.toLocaleString(), `held-out test games, ${score.postseason_games} of them in the playoffs`],
  [`1 of ${reg.n}`, "ideas that survived correction for trying so many"],
  [`${(best.roi * 100).toFixed(1)}%`, `average return per bet betting my picks against closing lines (${best.bets.toLocaleString()} bets)`],
  [String(ledger.rows.length), "live predictions locked into a tamper-evident ledger so far"],
].map(([n, l]) => h("div", { class: "stat" }, h("div", { class: "n" }, n), h("div", { class: "l" }, l))));

$("#lessons").append(...[
  ["🔍", "Honest measurement beats clever modeling", `Peeking at the future made a simple EPA model look better than the market. Honest versions score far worse. Every rule in this project exists to prevent that.`],
  ["📏", "Good, but the market is better", `My champion scores ${f4(champ.log_loss)} against the market's ${f4(mkt.log_loss)} (home-field-only: ${f4(home.log_loss)}). The ${f4(champ.gap_vs_market)} gap is statistically solid.`],
  ["🧮", `One idea in ${reg.n} survived`, "The Kalman rating filter (which remembers last season) was the only improvement that held up after correcting for how many things I tried."],
  ["💸", "No money in it", `Betting the model's picks at closing prices lost ${Math.abs(best.roi * 100).toFixed(1)}% per bet. Prices already contain my model's information, plus a margin.`],
  ["🔬", "Small samples cap what you can know", "With 111 playoff games, only a very large improvement is detectable. Saying 'I can't tell' is part of the result."],
].map(([icon, t, body]) => h("div", { class: "card" }, h("h3", {}, `${icon} ${t}`), h("p", { style: { margin: 0, color: "var(--muted)" } }, body))));

$("#foot-sha").textContent = meta.git_sha;
$("#foot-date").textContent = meta.built_utc;

// ---- demos (each module exports mount(el, data) and is isolated so one failure doesn't break the page)
// cache-bust demo modules: the build's git sha in production, a timestamp while developing locally
const bust = ["localhost", "127.0.0.1"].includes(location.hostname) ? Date.now() : meta.git_sha;
const DEMOS = ["quiz", "kalman", "qb", "leak", "walk", "score", "calib", "bracket", "replay", "vig", "bets", "hack", "registry", "power", "ledger", "title"];
for (const name of DEMOS) {
  const el = $(`#demo-${name}`);
  if (!el) continue;
  try {
    const mod = await import(`./demos/${name}.js?v=${bust}`);
    await mod.mount(el);
  } catch (err) {
    if (err instanceof TypeError && /Failed to fetch dynamically|Importing a module script failed|error loading dynamically/i.test(String(err.message))) {
      el.append(h("p", { class: "sub" }, "Demo coming soon."));
    } else {
      console.error(name, err);
      el.append(h("p", { class: "sub" }, `This demo failed to load (${err.message}).`));
    }
  }
}
