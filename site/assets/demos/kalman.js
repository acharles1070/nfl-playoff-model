import { h, s, loadJSON, plot, responsive, tooltip, pathFrom, slider, segmented, signed, f3 } from "../lib.js";

// Simplified 1-D Kalman filter on a team's per-game net EPA/play (offense minus defense allowed).
// The real model tracks offense, defense and each quarterback separately and adjusts for opponents.
function runFilter(games, priorMean, priorSd, obsSd, driftSd) {
  let x = priorMean, P = priorSd ** 2;
  const out = [];
  games.forEach((g, t) => {
    if (t > 0) P += driftSd ** 2;
    out.push({ x, sd: Math.sqrt(P) });              // belief BEFORE seeing this game
    const K = P / (P + obsSd ** 2);
    x += K * (g.obs - x);
    P *= 1 - K;
  });
  return { path: out, final: x, finalSd: Math.sqrt(P) };
}

export async function mount(el) {
  const data = await loadJSON("kalman");
  const st = { i: 0, obs: data.league_game_sd, drift: 0.03, prior: null, avg: true, model: true };
  const teamData = () => data.teams[st.i];
  st.prior = teamData().prior_sd;

  el.append(h("h3", {}, `📈 Watch a rating learn: real ${data.season} games`), h("p", { class: "sub" }, "Dots = how a team actually did each game (net EPA per play: its offense minus what its defense allowed). The green line is the running belief going into each game."));

  const top = h("div", { class: "row" });
  const holder = h("div", { class: "chart" });
  const readout = h("div", { class: "found", style: { marginTop: "10px" } });
  const sliders = h("div", { class: "row", style: { marginTop: "14px" } });
  el.append(top, holder, sliders, readout);

  const seg = segmented(data.teams.map((t, k) => [k, `${t.label}: ${t.team}`]), 0, (v) => { st.i = +v; st.prior = teamData().prior_sd; priorCtl.input.value = st.prior; priorCtl.out.textContent = f3(st.prior); redraw(); });
  top.append(seg);

  const mk = (o) => slider({ onInput: (v) => { o.set(v); redraw(); }, ...o.cfg });
  const obsCtl = mk({ set: (v) => (st.obs = v), cfg: { label: "How noisy is ONE game? (EPA/play)", min: 0.1, max: 0.9, step: 0.01, value: st.obs, fmt: f3 } });
  const driftCtl = mk({ set: (v) => (st.drift = v), cfg: { label: "How fast can a team really change per game?", min: 0, max: 0.12, step: 0.005, value: st.drift, fmt: f3 } });
  const priorCtl = mk({ set: (v) => (st.prior = v), cfg: { label: "How sure was I before week 1?", min: 0.02, max: 0.4, step: 0.01, value: st.prior, fmt: f3 } });
  sliders.append(obsCtl.el, driftCtl.el, priorCtl.el);

  const checks = h("div", { class: "row", style: { marginTop: "6px" } },
    ...[["avg", "Plain average so far"], ["model", "My full model's belief"]].map(([k, label]) =>
      h("label", { style: { fontSize: ".9rem" } }, h("input", { type: "checkbox", checked: true, onchange: (e) => { st[k] = e.target.checked; redraw(); } }), ` ${label}`)));
  el.append(checks);

  const tt = tooltip(holder);

  function redraw() {
    const t = teamData();
    const games = t.games;
    const f = runFilter(games, t.prior_mean, st.prior, st.obs, st.drift);
    const avgSoFar = games.map((_, k) => (k ? games.slice(0, k).reduce((a, g) => a + g.obs, 0) / k : t.prior_mean));
    const all = [...games.map((g) => g.obs), ...f.path.map((p) => p.x + 2 * p.sd), ...f.path.map((p) => p.x - 2 * p.sd)];
    const lo = Math.min(-0.5, ...games.map((g) => g.obs)) - 0.05, hi = Math.max(0.5, ...games.map((g) => g.obs)) + 0.05;

    responsive(holder, (w) => {
      const { svg, g, x, y } = plot(w, 300, { xDomain: [0.5, games.length + 0.5], yDomain: [lo, hi], xFmt: (v) => (Number.isInteger(v) ? v : ""), yFmt: (v) => signed(v, 1), xLabel: "game number", yLabel: "net EPA / play", xTicks: games.map((_, k) => k + 1).filter((v) => v % (w < 520 ? 4 : 2) === 1 || v === 1) });
      g.append(s("line", { x1: 0, x2: x(games.length + 0.5), y1: y(0), y2: y(0), stroke: "var(--muted)", opacity: 0.5 }));
      const band = f.path.map((p, k) => [x(k + 1), y(Math.min(hi, p.x + p.sd))]).concat(f.path.map((p, k) => [x(k + 1), y(Math.max(lo, p.x - p.sd))]).reverse());
      g.append(s("path", { d: pathFrom(band) + "Z", fill: "var(--model)", opacity: 0.18 }));
      if (st.avg) g.append(s("path", { d: pathFrom(avgSoFar.map((v, k) => [x(k + 1), y(v)])), fill: "none", stroke: "var(--blue)", "stroke-width": 2, "stroke-dasharray": "6 4" }));
      if (st.model) g.append(s("path", { d: pathFrom(games.map((q, k) => [x(k + 1), y(q.model)])), fill: "none", stroke: "var(--market)", "stroke-width": 2 }));
      g.append(s("path", { d: pathFrom(f.path.map((p, k) => [x(k + 1), y(p.x)])), fill: "none", stroke: "var(--model)", "stroke-width": 3 }));
      games.forEach((q, k) => {
        const dot = s("circle", { cx: x(k + 1), cy: y(q.obs), r: 5.5, fill: "var(--text)", opacity: 0.85, style: { cursor: "pointer" } });
        dot.addEventListener("pointermove", (ev) => tt.show(`<b>Week ${q.week} vs ${q.opp}</b><br>game: ${signed(q.obs, 2)}<br>belief going in: ${signed(f.path[k].x, 2)}`, ev));
        dot.addEventListener("pointerleave", () => tt.hide());
        g.append(dot);
      });
      return svg;
    });

    const avg = games.reduce((a, q) => a + q.obs, 0) / games.length;
    readout.replaceChildren(h("b", { class: "k" }, "Right now"),
      `After ${games.length} games the simplified filter rates ${t.team} at ${signed(f.final, 2)} (±${f3(f.finalSd)}); the plain average of their games is ${signed(avg, 2)}. `,
      st.obs > 0.6 ? "You're telling it single games are very noisy, so it barely moves off its starting belief." : st.obs < 0.2 ? "You're telling it single games are very reliable, so it chases every result." : "Try pushing the noise slider to each extreme.");
  }

  el.append(h("div", { class: "legend" },
    ...[["var(--text)", "game result"], ["var(--model)", "filter's belief going into each game (band = ±1 uncertainty)"], ["var(--blue)", "plain average so far"], ["var(--market)", "my full model (offense, defense and QB, opponent-adjusted)"]].map(([c, t]) => h("span", {}, h("i", { style: { background: c } }), t))));
  redraw();
}
