const $ = (s) => document.querySelector(s);
const picks = new Map();
const PALETTE = {
  "Action": ["#e5383b", "#5c0d12"], "Adventure": ["#f08a24", "#4a2106"], "Animation": ["#4cc9f0", "#173a7a"],
  "Children's": ["#ffd166", "#b5502c"], "Comedy": ["#ffbe0b", "#8a3b00"], "Crime": ["#5a5f73", "#0f1016"],
  "Documentary": ["#8ecae6", "#203a43"], "Drama": ["#9d4edd", "#240046"], "Fantasy": ["#7b2cbf", "#10002b"],
  "Film-Noir": ["#adb5bd", "#111"], "Horror": ["#8b0000", "#050505"], "Musical": ["#ff5d8f", "#4a0d2a"],
  "Mystery": ["#3a86ff", "#0b1d3a"], "Romance": ["#ff6b9a", "#5a0f2e"], "Sci-Fi": ["#3fc1b0", "#04202a"],
  "Thriller": ["#ef476f", "#1b0710"], "War": ["#8a9a5b", "#1f2414"], "Western": ["#d4a373", "#3b2412"],
};

function hash(n) { let x = (n + 1) * 2654435761; return (x >>> 0) % 360; }
function art(m) {
  const [a, b] = PALETTE[m.genres[0]] || ["#ffb547", "#2a1a08"];
  const c = (PALETTE[m.genres[1]] || [b])[0];
  const ang = hash(m.id) % 180;
  return `background:radial-gradient(120% 80% at ${20 + (m.id % 60)}% 0%, ${a}, transparent 60%),
    linear-gradient(${ang}deg, ${b}, ${c}55 70%, ${b})`;
}
function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

function poster(m, { pickable = false, rank = null, delay = 0 } = {}) {
  const el = document.createElement(pickable ? "button" : "div");
  el.className = "poster" + (picks.has(m.id) ? " on" : "");
  el.style.animationDelay = `${delay}ms`;
  el.innerHTML = `<div class="art" style="${art(m)}"></div>
    <div class="glyph">${esc(m.title.replace(/^(The|A|An) /, "")[0] || "")}</div>
    ${pickable ? '<span class="tick">✓</span>' : ""}${rank ? `<span class="rank">${rank}</span>` : ""}
    <div class="meta"><div class="t">${esc(m.title)}</div><div class="y">${m.year} · ${esc(m.genres.slice(0, 2).join(" / "))}</div></div>`;
  if (pickable) {
    el.setAttribute("aria-pressed", picks.has(m.id));
    el.addEventListener("click", () => { toggle(m); el.classList.toggle("on", picks.has(m.id)); el.setAttribute("aria-pressed", picks.has(m.id)); });
  }
  return el;
}

function toggle(m) { picks.has(m.id) ? picks.delete(m.id) : picks.set(m.id, m); renderStub(); }
function renderStub() {
  const ol = $("#stub-list");
  ol.innerHTML = "";
  for (const m of picks.values()) {
    const li = document.createElement("li");
    li.innerHTML = `<span>${esc(m.title)} <small style="color:#8a7f70">${m.year}</small></span><button aria-label="remove">✕</button>`;
    li.querySelector("button").onclick = () => { picks.delete(m.id); renderStub(); syncGrid(); };
    ol.appendChild(li);
  }
  $("#stub-count").textContent = `${picks.size} picked`;
  $("#stub-empty").hidden = picks.size > 0;
}
function syncGrid() { document.querySelectorAll("#pick-grid .poster").forEach((p) => p.classList.toggle("on", picks.has(+p.dataset.id))); }

async function api(path, opts) { const r = await fetch(path, opts); if (!r.ok) throw new Error(r.status); return r.json(); }

async function loadGrid(q = "") {
  const { results } = await api(q ? `/api/movies?q=${encodeURIComponent(q)}&limit=30` : "/api/popular?limit=30");
  const g = $("#pick-grid");
  g.innerHTML = "";
  results.forEach((m, i) => { const p = poster(m, { pickable: true, delay: i * 18 }); p.dataset.id = m.id; g.appendChild(p); });
}

function showResults(data, title) {
  $("#results").hidden = false;
  $("#res-title").textContent = title;
  const t = data.timing;
  $("#trace").innerHTML = `
    <div><b>${t.catalog.toLocaleString()}</b><span>films in catalog</span></div>
    <div><b>${t.candidates}</b><span>candidates · ${t.retrieve_ms.toFixed(1)} ms</span></div>
    <div><b>${data.recommendations.length}</b><span>ranked · ${t.rank_ms.toFixed(1)} ms</span></div>
    <div class="hot"><b>${t.total_ms.toFixed(1)} ms</b><span>end to end</span></div>`;
  const h = $("#history");
  if (data.history) {
    h.hidden = false;
    h.innerHTML = `<span class="lab">Viewer #${data.user} · ${data.history_size} ratings · recently watched</span>` +
      data.history.slice(0, 8).map((m) => `<span class="pill">${esc(m.title)}</span>`).join("");
  } else h.hidden = true;
  const g = $("#rec-grid");
  g.innerHTML = "";
  data.recommendations.forEach((m, i) => {
    const wrap = document.createElement("div");
    wrap.className = "rec";
    wrap.style.animationDelay = `${i * 45}ms`;
    wrap.appendChild(poster(m, { rank: m.rank, delay: i * 45 }));
    const why = m.because ? `Because you liked <b>${esc(m.because.title)}</b>` : "Popular with viewers like you";
    const tags = m.sources.map((s) => `<span class="tag ${s === "two-tower" ? "tt" : s === "ALS" ? "als" : "pop"}">${s}</span>`).join("");
    wrap.insertAdjacentHTML("beforeend", `<div class="why">${why}</div><div class="tags">${tags}<span class="tag">genre ${(m.genre_affinity * 100).toFixed(0)}%</span></div>`);
    g.appendChild(wrap);
  });
  $("#results").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function recommend() {
  const body = { liked: [...picks.keys()], k: 12 };
  if ($("#age").value !== "") body.age = +$("#age").value;
  if ($("#gender").value !== "") body.gender = +$("#gender").value;
  const data = await api("/api/recommend", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const first = data.liked[0];
  showResults(data, data.cold_start ? "Cold start: no picks yet" : picks.size === 1 ? `Because you liked ${first.title}` : "Because you liked…");
}

async function randomViewer() {
  const { users } = await api("/api/users?n=1");
  const data = await api(`/api/users/${users[0]}?k=12`);
  showResults(data, `Up next for viewer #${data.user}`);
}

function pct(a, b) { return `${a >= b ? "+" : ""}${(((a - b) / b) * 100).toFixed(0)}%`; }

async function loadStats() {
  const s = await api("/api/stats");
  const off = s.offline || {};
  const R = off.results || {};
  const best = R["Two-stage (two-tower + LightGBM)"], pop = R["Popularity"];
  if (best && pop) $("#c-lift").textContent = pct(best["ndcg@10"], pop["ndcg@10"]);
  if (s.benchmark) $("#c-p95").textContent = `${s.benchmark.p95_ms.toFixed(0)} ms`;
  $("#c-cand").textContent = "~220";
  const h = await api("/api/health");
  $("#c-cat").textContent = h.items.toLocaleString();
  if (off.split) $("#split-note").textContent = `Temporal split: train before ${off.split.t_val}, validate until ${off.split.t_test}, test after. ` +
    `${off.warm_users.toLocaleString()} test users with history; target = films rated 4+ that they had not seen.`;
  const cols = ["recall@10", "ndcg@10", "map@10", "recall@50", "coverage@10", "novelty@10"];
  const maxN = Math.max(...Object.values(R).map((r) => r["ndcg@10"]));
  const rows = Object.entries(R).map(([k, r]) => `<tr class="${r === best ? "best" : ""}"><td>${esc(k)}</td>` +
    cols.map((c) => c === "ndcg@10"
      ? `<td><span class="bar" style="width:${(r[c] / maxN) * 80}px"></span>${r[c].toFixed(4)}</td>`
      : `<td>${c === "coverage@10" ? (r[c] * 100).toFixed(1) + "%" : r[c].toFixed(c === "novelty@10" ? 2 : 4)}</td>`).join("") + "</tr>").join("");
  $("#mtable").innerHTML = `<thead><tr><th>Model</th>${cols.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
}

function bulbs() {
  document.querySelectorAll(".bulbs").forEach((b) => {
    const horiz = b.classList.contains("top") || b.classList.contains("bottom");
    const n = Math.max(6, Math.floor((horiz ? b.offsetWidth : b.offsetHeight) / 22));
    b.innerHTML = "<i></i>".repeat(n);
  });
}

let timer;
$("#q").addEventListener("input", (e) => { clearTimeout(timer); timer = setTimeout(() => loadGrid(e.target.value), 160); });
$("#go").addEventListener("click", recommend);
$("#clear").addEventListener("click", () => { picks.clear(); renderStub(); syncGrid(); });
$("#hero-random").addEventListener("click", randomViewer);
window.addEventListener("resize", bulbs);

(async () => {
  bulbs();
  renderStub();
  await loadGrid();
  loadStats().catch(console.error);
  // seed with a couple of well-known picks so the page opens on a real result
  const seed = await api("/api/movies?q=matrix&limit=1");
  const seed2 = await api("/api/movies?q=toy story&limit=1");
  [...seed.results, ...seed2.results].forEach((m) => picks.set(m.id, m));
  renderStub(); syncGrid();
  if (new URLSearchParams(location.search).has("demo")) recommend();
})();
