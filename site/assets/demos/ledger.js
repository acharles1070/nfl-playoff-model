import { h, loadJSON, pct, f3, logLoss1 } from "../lib.js";

async function sha256(str) {
  const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(str));
  return [...new Uint8Array(buf)].map((b) => b.toString(16).padStart(2, "0")).join("");
}

export async function mount(el) {
  const d = await loadJSON("ledger");
  const meta = await loadJSON("meta");
  const orig = d.rows.map((r) => ({ ...r }));
  const bodyOf = (row) => d.fields.filter((k) => k !== "row_hash").map((k) => `${k}=${row[k]}`).join("|");
  const hashOf = (prev, row) => sha256(`${prev}|${bodyOf(row)}`);
  let work = orig.map((r) => ({ ...r })), k = 3, newP = "0.99", msg = "";
  const publishedHead = orig[orig.length - 1].row_hash;

  const results = (() => {
    const by = {};
    d.rows.forEach((r) => {
      const f = d.finished[r.game_id]; if (!f || f.home_score === f.away_score) return;
      const y = f.home_score > f.away_score ? 1 : 0;
      const o = (by[r.model_id] ||= { n: 0, m: 0, k: 0 });
      o.n++; o.m += logLoss1(+r.p_home_win, y); o.k += logLoss1(+r.market_p_home, y);
    });
    return by;
  })();

  el.append(h("h3", {}, "🔐 The ledger: try to cheat it"), h("p", { class: "sub" }, `${d.rows.length} real predictions logged before kickoff (SHA-256 computed live in your browser). Each row's fingerprint covers its own contents plus the previous row's fingerprint.`));
  const table = h("div", { class: "scroll", style: { maxHeight: "310px", overflowY: "auto", border: "1px solid var(--line)", borderRadius: "10px" } });
  const controls = h("div", { class: "row", style: { marginTop: "12px" } });
  const status = h("div", { class: "found", style: { marginTop: "12px" } });
  const resultBox = h("div", { class: "card", style: { marginTop: "12px" } });
  el.append(table, controls, status, resultBox);

  const rowSel = h("select", { style: selStyle(), onchange: (e) => { k = +e.target.value; } }, ...orig.slice(0, 12).map((r, i) => h("option", { value: i + 1, selected: i + 1 === k || null }, `#${i + 1} ${r.away_team}@${r.home_team}`)));
  const pIn = h("input", { type: "number", min: 0, max: 1, step: 0.01, value: newP, style: { ...selStyle(), width: "90px" }, oninput: (e) => { newP = e.target.value; } });
  controls.append(h("label", {}, "Row ", rowSel), h("label", {}, "change my p(home win) to ", pIn),
    h("button", { class: "btn primary", onclick: async () => { work[k - 1].p_home_win = (+newP).toFixed(8); msg = "tamper"; await refresh(); } }, "1. Tamper"),
    h("button", { class: "btn", onclick: async () => { work[k - 1].row_hash = await hashOf(work[k - 1].prev_hash, work[k - 1]); msg = "resign1"; await refresh(); } }, "2. Re-sign that row"),
    h("button", { class: "btn", onclick: async () => { for (let i = k; i < work.length; i++) { work[i].prev_hash = work[i - 1].row_hash; work[i].row_hash = await hashOf(work[i].prev_hash, work[i]); } msg = "resignAll"; await refresh(); } }, "3. Re-sign everything after it"),
    h("button", { class: "btn", onclick: async () => { work = orig.map((r) => ({ ...r })); msg = ""; await refresh(); } }, "Reset"));

  async function refresh() {
    let prev = d.genesis; const st = [];
    for (const r of work) {
      const linkOK = r.prev_hash === prev, selfOK = (await hashOf(prev, r)) === r.row_hash;
      st.push(linkOK && selfOK ? "ok" : !selfOK ? "edited" : "broken"); prev = r.row_hash;
    }
    const head = work[work.length - 1].row_hash;
    table.replaceChildren(h("table", {}, h("thead", {}, h("tr", {}, ...["#", "Game", "My p(home)", "Market p(home)", "Result", "Fingerprint", "Check"].map((t, i) => h("th", { class: i >= 2 && i <= 3 ? "num" : "" }, t)))),
      h("tbody", {}, ...work.map((r, i) => {
        const f = d.finished[r.game_id], tampered = r.p_home_win !== orig[i].p_home_win;
        return h("tr", { style: { background: st[i] === "ok" ? "" : "color-mix(in srgb, var(--bad) 18%, transparent)" } },
          h("td", {}, r.entry_id), h("td", {}, `${r.away_team} @ ${r.home_team} (wk ${r.week})`),
          h("td", { class: "num", style: { fontWeight: tampered ? 800 : 400, color: tampered ? "var(--bad)" : "" } }, (+r.p_home_win).toFixed(3)), h("td", { class: "num" }, (+r.market_p_home).toFixed(3)),
          h("td", {}, f ? `${f.away_score}–${f.home_score}` : "pending"), h("td", { class: "mono" }, `${r.row_hash.slice(0, 10)}…`),
          h("td", {}, st[i] === "ok" ? "✓" : st[i] === "edited" ? "✗ row edited" : "✗ chain broken"));
      }))));
    const bad = st.filter((x) => x !== "ok").length;
    const text = {
      "": "All rows verify. This is the real ledger. Try step 1.",
      tamper: `You changed row ${k}. Its stored fingerprint no longer matches its contents: caught immediately.`,
      resign1: `You recomputed row ${k}'s fingerprint, so that row looks fine, but the next row stored the OLD fingerprint, so the chain breaks right after it.`,
      resignAll: "You recomputed every row after it, so the whole chain verifies again. But look at the head fingerprint below: it no longer matches the one I published and committed to git before the games were played. Quiet edits leave a fingerprint.",
    }[msg];
    status.replaceChildren(h("b", { class: "k" }, bad ? `${bad} row${bad === 1 ? "" : "s"} fail verification` : "Chain intact ✓"), text,
      h("div", { class: "mono", style: { fontSize: ".78rem", marginTop: "8px", wordBreak: "break-all" } }, `head fingerprint now:       ${head}`, h("br"), `head fingerprint published: ${publishedHead}`, h("br"), head === publishedHead ? "✓ same" : "✗ DIFFERENT: tampering exposed"));
  }

  const models = Object.entries(results);
  resultBox.append(h("b", {}, "How have the live predictions done so far?"), models.length ? h("div", { class: "scroll" }, h("table", { style: { marginTop: "8px" } }, h("thead", {}, h("tr", {}, ...["Model", "Games finished", "My log loss", "Market log loss (line when logged)"].map((t, i) => h("th", { class: i ? "num" : "" }, t)))),
    h("tbody", {}, ...models.map(([id, o]) => h("tr", {}, h("td", { class: "mono" }, id), h("td", { class: "num" }, o.n), h("td", { class: "num" }, f3(o.m / o.n)), h("td", { class: "num" }, f3(o.k / o.n))))))) : h("p", { class: "sub" }, "No games finished yet."),
    h("p", { class: "sub", style: { margin: "8px 0 0" } }, `With only a handful of games, differences between the model and the market are pure noise (see Chapter 12). The point of the ledger is the record I'm building: ${d.rows.length} predictions so far, and 2026 will add roughly 285 more per model.`));
  await refresh();
}

const selStyle = () => ({ padding: "8px", borderRadius: "10px", border: "1px solid var(--line)", background: "var(--panel2)", color: "var(--text)", font: "inherit" });
