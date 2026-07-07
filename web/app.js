// TokenOptimizer cockpit — polls the proxy and drives the live UI.
const $ = (id) => document.getElementById(id);
const money = (v) => "$" + (v ?? 0).toFixed(6);
const money2 = (v) => "$" + (v ?? 0).toFixed(2);
const num = (v) => (v ?? 0).toLocaleString();

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(r.status);
  return r.json();
}

function pill(route) {
  return `<span class="pill ${route}">${route}</span>`;
}

async function pollStats() {
  try {
    const s = await getJSON("/api/stats");
    $("saved-usd").textContent = money(s.saved_usd);
    $("saved-pct").textContent = s.saved_pct + "%";
    $("tokens-avoided").textContent = num(s.remote_tokens_avoided);
    $("baseline-usd").textContent = money2(s.baseline_usd);
    $("spent-usd").textContent = money2(s.spent_usd);
    $("total-req").textContent = num(s.total_requests);
    $("avg-lat").textContent = Math.round(s.avg_latency_ms) + " ms";

    $("mode-badge").textContent = "mode: " + s.mode;
    $("embed-badge").textContent = "embedder: " + s.embedder;
    $("models").textContent = `local ${s.models.local}  ·  remote ${s.models.remote}`;

    const rc = s.route_counts;
    const total = Math.max(1, s.total_requests);
    $("seg-cache").style.width = (rc.cache / total) * 100 + "%";
    $("seg-local").style.width = (rc.local / total) * 100 + "%";
    $("seg-remote").style.width = (rc.remote / total) * 100 + "%";
    $("c-cache").textContent = rc.cache;
    $("c-local").textContent = rc.local;
    $("c-remote").textContent = rc.remote;
    $("local-pct").textContent = s.local_pct + "% offloaded";
    $("cache-hitrate").textContent = Math.round(s.cache.hit_rate * 100) + "%";
    $("cache-size").textContent = s.cache.size;
  } catch (e) { /* proxy warming up */ }
}

async function pollGpu() {
  try {
    const g = await getJSON("/api/gpu");
    $("gpu-name").textContent = g.name;
    $("gpu-src").textContent = g.source;
    $("gpu-util").textContent = Math.round(g.util_percent) + "%";
    $("gpu-fill").style.width = g.util_percent + "%";
    $("gpu-mem").textContent = `${(g.mem_used_mb / 1024).toFixed(1)} / ${(g.mem_total_mb / 1024).toFixed(0)} GB`;
    $("gpu-temp").textContent = Math.round(g.temp_c) + " °C";
    $("gpu-power").textContent = Math.round(g.power_w) + " W";

    const hot = (g.activity ?? 0) > 0.15;
    $("gpu-fill").parentElement.parentElement.parentElement.classList.toggle("hot", hot);
    $("gpu-hint").textContent = hot
      ? "◉ on-device inference — answering locally, 0 tokens to cloud"
      : "idle — waiting for an on-device query";
  } catch (e) { /* ignore */ }
}

function timeAgo(ts) {
  const s = Math.max(0, Date.now() / 1000 - ts);
  if (s < 60) return Math.floor(s) + "s";
  return Math.floor(s / 60) + "m";
}

async function pollRecent() {
  try {
    const { records } = await getJSON("/api/recent");
    $("recent-body").innerHTML = records.map((r) => `
      <tr>
        <td>${pill(r.route)}</td>
        <td class="qcell" title="${escapeHtml(r.query)}">${escapeHtml(r.query)}</td>
        <td>${r.prompt_tokens + r.completion_tokens}</td>
        <td>${Math.round(r.latency_ms)} ms</td>
        <td>${money(r.cost_usd)}</td>
        <td class="save-pos">${r.saved_usd > 0 ? "+" + money(r.saved_usd) : "—"}</td>
        <td class="why" title="${escapeHtml(r.reason || "")}">${escapeHtml(r.reason || "")}</td>
      </tr>`).join("");
  } catch (e) { /* ignore */ }
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

async function send(query) {
  if (!query.trim()) return;
  const btn = $("send");
  btn.disabled = true;
  $("answer").innerHTML = '<span class="muted">routing…</span>';
  try {
    const data = await getJSON("/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: query }] }),
    });
    const x = data.x_tokenoptimizer || {};
    const text = data.choices?.[0]?.message?.content || "";
    $("answer").innerHTML =
      `<span class="verdict ${x.route}">${x.route.toUpperCase()}${x.complexity != null ? " · complexity " + x.complexity : ""}${x.cache_score != null ? " · sim " + x.cache_score : ""}</span>` +
      `<div>${escapeHtml(text)}</div>` +
      `<div class="muted" style="margin-top:8px">saved ${money(x.saved_usd)} vs baseline ${money(x.baseline_usd)} · ${escapeHtml(x.reason || "")}</div>`;
    // refresh immediately so the GPU spike + counters feel instant
    pollGpu(); pollStats(); pollRecent();
  } catch (e) {
    $("answer").innerHTML = '<span class="muted">error: ' + e.message + "</span>";
  } finally {
    btn.disabled = false;
  }
}

$("send").addEventListener("click", () => send($("q").value));
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") send($("q").value); });
document.querySelectorAll(".chip").forEach((c) =>
  c.addEventListener("click", () => { $("q").value = c.dataset.q; send(c.dataset.q); })
);

// fast GPU poll for a lively gauge; slower stats/feed
setInterval(pollGpu, 500);
setInterval(pollStats, 1000);
setInterval(pollRecent, 1500);
pollGpu(); pollStats(); pollRecent();
