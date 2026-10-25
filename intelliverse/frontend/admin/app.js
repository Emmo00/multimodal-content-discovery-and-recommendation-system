const state = { view: "overview", health: null, error: null, busy: false };
const app = document.querySelector("#app");
const toast = document.querySelector("#toast");

async function request(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}
function notify(message) { toast.textContent = message; toast.classList.add("visible"); setTimeout(() => toast.classList.remove("visible"), 2600); }
function setNav() { document.querySelectorAll(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.view === state.view)); }
function render() { setNav(); if (state.view === "discovery") return discovery(); if (state.view === "content") return content(); if (state.view === "evaluation") return evaluation(); return overview(); }

async function overview() {
  app.innerHTML = `<p class="tag">OPERATIONS / OVERVIEW</p><h1>System instrument panel.</h1><p>Administrative controls are deliberately separated from the user experience.</p><div class="empty">Loading system status...</div>`;
  try { state.health = await request("/admin/api/health"); }
  catch (error) { return showError(error); }
  const h = state.health;
  app.innerHTML = `<p class="tag">OPERATIONS / OVERVIEW</p><h1>System instrument panel.</h1><div class="grid">
    <div class="panel metric"><span class="tag">STATUS</span><strong class="status">${h.status}</strong></div>
    <div class="panel metric"><span class="tag">INDEXED CONTENT</span><strong>${h.corpus_size}</strong></div>
    <div class="panel metric"><span class="tag">EMBEDDING DIMENSION</span><strong>${h.embedding_dimension}</strong></div>
    <div class="panel metric"><span class="tag">SOURCES</span><strong>${h.source_count}</strong></div>
  </div><section class="panel"><h2>Runtime</h2><p>Model: ${escapeHtml(h.embedding_model)}</p><p>Storage: ${escapeHtml(h.storage_backend)}${h.in_memory ? " (session data resets on restart)" : ""}</p><p>Retrieval: top ${h.settings.top_k} · result floor ${h.settings.min_result_similarity}</p></section>`;
}

function discovery() {
  app.innerHTML = `<p class="tag">OPERATIONS / DISCOVERY</p><h1>Grow the index.</h1><p>Query every configured source, remove duplicate URLs, embed new metadata, and add it to the vector index.</p>
    <form id="discover-form" class="form"><input name="query" placeholder="Topic to discover" required><input name="limit" type="number" min="1" max="100" value="25"><button class="pill-button primary" type="submit">${state.busy ? "Running..." : "Discover and index"}</button></form><div id="discovery-result"></div>`;
  document.querySelector("#discover-form").addEventListener("submit", runDiscovery);
}
async function runDiscovery(event) {
  event.preventDefault(); state.busy = true; discovery();
  const form = new FormData(event.currentTarget);
  try { const params = new URLSearchParams({ q: form.get("query"), limit: form.get("limit") }); const result = await request(`/admin/api/discover?${params}`, { method: "POST" }); document.querySelector("#discovery-result").innerHTML = `<div class="notice">Retrieved ${result.retrieved} · indexed ${result.indexed} · duplicates ${result.duplicates} · corpus size ${result.corpus_size}</div>`; notify("Discovery complete"); }
  catch (error) { document.querySelector("#discovery-result").innerHTML = `<div class="panel error">${escapeHtml(error.message)}</div>`; }
  finally { state.busy = false; }
}

async function content() {
  app.innerHTML = `<p class="tag">OPERATIONS / CONTENT</p><h1>Indexed catalogue.</h1><form id="content-form" class="form"><input name="query" placeholder="Filter by title, description or topic"><select name="type"><option value="">All types</option><option>video</option><option>audio</option><option>text</option></select><button class="pill-button primary" type="submit">Refresh</button></form><div id="content-table" class="empty">Loading catalogue...</div>`;
  document.querySelector("#content-form").addEventListener("submit", (event) => { event.preventDefault(); loadContent(new FormData(event.currentTarget)); });
  loadContent();
}
async function loadContent(form) {
  const params = new URLSearchParams({ limit: "200" }); if (form?.get("query")) params.set("query", form.get("query")); if (form?.get("type")) params.set("content_type", form.get("type"));
  try { const data = await request(`/admin/api/content?${params}`); document.querySelector("#content-table").innerHTML = data.items.length ? `<div class="table-wrap"><table><thead><tr><th>Title</th><th>Type</th><th>Source</th><th>Discovered</th></tr></thead><tbody>${data.items.map((item) => `<tr><td><a class="back-link" href="${escapeAttribute(item.source_url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title)}</a></td><td>${item.content_type}</td><td>${escapeHtml(item.source_platform)}</td><td>${item.discovered_at ? new Date(item.discovered_at).toLocaleDateString() : "-"}</td></tr>`).join("")}</tbody></table></div>` : "No indexed content."; } catch (error) { showError(error); }
}

function evaluation() {
  app.innerHTML = `<p class="tag">OPERATIONS / EVALUATION</p><h1>Compare ranking modes.</h1><p>Discovery is disabled so both rankings use the same fixed corpus.</p><form id="evaluation-form" class="form"><input name="query" placeholder="Query to compare" required><input name="user_id" placeholder="Optional user ID"><button class="pill-button primary" type="submit">Compare</button></form><div id="evaluation-result"></div>`;
  document.querySelector("#evaluation-form").addEventListener("submit", runEvaluation);
}
async function runEvaluation(event) {
  event.preventDefault(); const form = new FormData(event.currentTarget); const params = new URLSearchParams({ q: form.get("query"), limit: "5" }); if (form.get("user_id")) params.set("user_id", form.get("user_id"));
  try { const data = await request(`/admin/api/evaluation?${params}`); document.querySelector("#evaluation-result").innerHTML = `<div class="compare">${comparison("Semantic only", data.semantic_only)}${comparison("Personalised", data.personalised)}</div>`; } catch (error) { showError(error); }
}
function comparison(title, results) { return `<section class="panel"><h2>${title}</h2>${results.length ? results.map((item, index) => `<div class="result-line">${index + 1}. ${escapeHtml(item.title)}<small>score ${item.score} · semantic ${item.semantic_similarity} · profile ${item.profile_similarity}</small></div>`).join("") : `<p>No results.</p>`}</section>`; }
function showError(error) { app.innerHTML = `<div class="panel error">${escapeHtml(error.message)}</div>`; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char])); }
function escapeAttribute(value) { return escapeHtml(value); }
document.querySelectorAll(".nav-link").forEach((button) => button.addEventListener("click", () => { state.view = button.dataset.view; render(); }));
render();
