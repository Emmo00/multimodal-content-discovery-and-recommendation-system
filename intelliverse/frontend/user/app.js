const state = {
  view: "discover",
  user: JSON.parse(localStorage.getItem("intelliverse-user") || "null"),
  query: "",
  type: "",
  personalised: true,
  discovery: true,
  results: null,
  profile: null,
  busy: false,
  error: null,
};

const app = document.querySelector("#app");
const toast = document.querySelector("#toast");

async function request(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
  return data;
}

function notify(message) {
  toast.textContent = message;
  toast.classList.add("visible");
  setTimeout(() => toast.classList.remove("visible"), 2600);
}

function render() {
  document.querySelector("#user-chip").textContent = state.user ? state.user.name : "Guest";
  document.querySelectorAll(".nav-link").forEach((button) => button.classList.toggle("active", button.dataset.view === state.view));
  if (!state.user) return renderRegistration();
  if (state.view === "profile") return renderProfile();
  return renderDiscover();
}

function renderRegistration() {
  app.innerHTML = `<section class="panel modal">
    <p class="tag">WELCOME TO INTELLIVERSE</p><h1>Find something worth your attention.</h1>
    <p>Create a local demo profile to enable feedback and personalised ranking.</p>
    <form id="register-form">
      <label class="field">Name<input name="name" required maxlength="255" autocomplete="name"></label>
      <label class="field">Email<input name="email" type="email" required autocomplete="email"></label>
      <button class="pill-button primary" type="submit">Start exploring</button>
    </form>
  </section>`;
  document.querySelector("#register-form").addEventListener("submit", register);
}

async function register(event) {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  try {
    const user = await request("/api/users", { method: "POST", body: JSON.stringify({ name: form.get("name"), email: form.get("email") }) });
    state.user = user;
    localStorage.setItem("intelliverse-user", JSON.stringify(user));
    notify("Profile created");
    render();
  } catch (error) { state.error = error.message; renderError(); }
}

function renderDiscover() {
  const response = state.results;
  app.innerHTML = `<section class="hero">
    <p class="tag">SEMANTIC DISCOVERY / PERSONAL RANKING</p>
    <h1>Find something worth your attention.</h1>
    <p>Search across video, audio and text. Your feedback makes the next ranking more useful.</p>
    <form id="search-form" class="search-form">
      <input name="query" value="${escapeHtml(state.query)}" placeholder="What would you like to learn about?" required>
      <button class="pill-button primary" type="submit">${state.busy ? "Searching..." : "Search"}</button>
    </form>
    <div class="filters" aria-label="Search filters">
      ${["", "video", "audio", "text"].map((type) => `<button type="button" class="filter ${state.type === type ? "active" : ""}" data-type="${type}">${type || "All"}</button>`).join("")}
      <label class="toggle"><input id="personalised" type="checkbox" ${state.personalised ? "checked" : ""}> Personalised</label>
      <label class="toggle"><input id="discovery" type="checkbox" ${state.discovery ? "checked" : ""}> Discover when needed</label>
    </div>
  </section>
  ${state.error ? `<div class="error panel">${escapeHtml(state.error)}</div>` : ""}
  ${response ? renderResults(response) : `<div class="empty">Try a query to retrieve and rank content.</div>`}`;
  document.querySelector("#search-form").addEventListener("submit", search);
  document.querySelectorAll("[data-type]").forEach((button) => button.addEventListener("click", () => { state.type = button.dataset.type; render(); }));
  document.querySelector("#personalised").addEventListener("change", (event) => { state.personalised = event.target.checked; });
  document.querySelector("#discovery").addEventListener("change", (event) => { state.discovery = event.target.checked; });
  document.querySelectorAll("[data-action]").forEach((button) => button.addEventListener("click", () => interact(button.dataset.id, button.dataset.action, button)));
}

function renderResults(response) {
  const notice = response.discovery_triggered ? `<div class="notice">${response.newly_indexed ? `${response.newly_indexed} new items discovered and indexed.` : "Discovery ran, but no new items were added."}</div>` : "";
  if (!response.results.length) return `${notice}<div class="empty">No relevant results found.</div>`;
  return `${notice}<p class="tag">${response.candidates_considered} candidates considered · ${response.personalised ? "personalised ranking" : "semantic ranking"}</p><section class="results">${response.results.map(renderResult).join("")}</section>`;
}

function renderResult(result) {
  return `<article class="result-card">
    <div class="result-head"><span class="tag">${result.content_type.toUpperCase()} · ${escapeHtml(result.source_platform)}</span><span class="score">Score ${result.score.toFixed(3)}</span></div>
    <h3>${escapeHtml(result.title)}</h3><p>${escapeHtml(result.description)}</p>
    <div class="result-meta">${result.topics.map((topic) => `<span class="tag">#${escapeHtml(topic)}</span>`).join("")}</div>
    <details><summary>Why this result?</summary><p>Semantic ${result.semantic_similarity.toFixed(3)} · Profile ${result.profile_similarity.toFixed(3)} · Freshness ${result.freshness.toFixed(3)}</p></details>
    <div class="actions"><a class="source-link" href="${escapeAttribute(result.source_url)}" target="_blank" rel="noopener noreferrer">Open original</a>
      ${["like", "save", "dislike", "skip"].map((action) => `<button type="button" data-action="${action}" data-id="${result.content_id}">${action}</button>`).join("")}
    </div>
  </article>`;
}

async function search(event) {
  event.preventDefault();
  state.query = new FormData(event.currentTarget).get("query").trim();
  state.busy = true; state.error = null; render();
  const params = new URLSearchParams({ q: state.query, user_id: state.user.user_id, personalise: String(state.personalised), allow_discovery: String(state.discovery), limit: "10" });
  if (state.type) params.append("content_type", state.type);
  try { state.results = await request(`/api/search?${params}`); }
  catch (error) { state.error = error.message; }
  finally { state.busy = false; render(); }
}

async function interact(contentId, action, button) {
  button.disabled = true;
  try { const result = await request("/api/interactions", { method: "POST", body: JSON.stringify({ user_id: state.user.user_id, content_id: contentId, interaction_type: action }) }); button.classList.add("selected"); notify(`${action} recorded · ${result.interaction_count} interactions`); }
  catch (error) { button.disabled = false; notify(error.message); }
}

async function renderProfile() {
  app.innerHTML = `<section class="hero"><p class="tag">YOUR PROFILE</p><h1>What the system has learned.</h1><p>Feedback is folded into your interest profile and used to reorder future results.</p></section><div class="empty">Loading profile...</div>`;
  try { state.profile = await request(`/api/users/${state.user.user_id}/profile`); }
  catch (error) { state.error = error.message; return renderError(); }
  app.innerHTML = `<section class="hero"><p class="tag">YOUR PROFILE</p><h1>${escapeHtml(state.user.name)}</h1><p>${escapeHtml(state.user.email)} · ${state.profile.personalisable ? "Personalisation active" : "More interactions needed for personalisation"}</p></section>
    <section class="profile-grid"><div class="panel metric"><span class="tag">INTERACTIONS</span><strong>${state.profile.interaction_count}</strong></div><div class="panel metric"><span class="tag">STATUS</span><strong>${state.profile.personalisable ? "Active" : "Cold start"}</strong></div><div class="panel metric"><span class="tag">UPDATED</span><strong>${new Date(state.profile.last_updated).toLocaleDateString()}</strong></div></section>
    <section class="panel"><h2>Content preferences</h2>${Object.entries(state.profile.content_type_preferences).map(([type, value]) => `<div class="metric-row"><span>${type}</span><span>${Math.round(value * 100)}%</span></div><div class="bar"><span style="width:${Math.round(value * 100)}%"></span></div>`).join("")}</section>`;
}

function renderError() { app.innerHTML = `<div class="error panel">${escapeHtml(state.error || "Something went wrong.")}</div>`; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#039;" }[char])); }
function escapeAttribute(value) { return escapeHtml(value); }

document.querySelectorAll(".nav-link").forEach((button) => button.addEventListener("click", () => { state.view = button.dataset.view; state.error = null; render(); }));
render();
