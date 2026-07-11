const resultsDiv = document.getElementById("results");
const recommendBtn = document.getElementById("recommendBtn");
const trendingBtn = document.getElementById("trendingBtn");
const healthBtn = document.getElementById("healthBtn");
const zatchRecommendBtn = document.getElementById("zatchRecommendBtn");
const zatchHealthBtn = document.getElementById("zatchHealthBtn");

function setLoading(message) {
  resultsDiv.innerHTML = `<div class="panel"><h2>${message}</h2></div>`;
}

function setError(message) {
  resultsDiv.innerHTML = `<div class="panel error"><h2>${message}</h2></div>`;
}

function renderReelTable(title, data) {
  if (!data.recommendations || data.recommendations.length === 0) {
    setError("No recommendations found.");
    return;
  }

  const rows = data.recommendations
    .map((item, index) => {
      const details = [
        item.type,
        item.products && item.products.length ? `${item.products.length} products` : "",
        item.hashtags && item.hashtags.length ? `tags: ${item.hashtags.slice(0, 3).join(", ")}` : "",
      ].filter(Boolean).join(" - ");

      return `
        <tr>
          <td>${index + 1}</td>
          <td><strong>${item.id}</strong><span>${details}</span></td>
          <td>${item.normalized_score}</td>
          <td>${item.score}</td>
          <td>${item.reason}</td>
        </tr>
      `;
    })
    .join("");

  resultsDiv.innerHTML = `
    <div class="panel">
      <h2>${title}</h2>
      <p>Strategy: ${data.strategy}. Returned ${data.count} live reels/bits.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Reel</th>
              <th>Normalized</th>
              <th>Raw Score</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
    </div>
  `;
}

function apiKeyHeaders() {
  const key = document.getElementById("api_key").value.trim();
  return key ? { "X-API-Key": key } : {};
}

async function requestJson(url) {
  const response = await fetch(url, { headers: apiKeyHeaders() });
  const data = await response.json();

  if (!response.ok || data.status !== "success") {
    throw new Error(data.message || "Request failed");
  }

  return data;
}

async function getRecommendations() {
  const userId = document.getElementById("user_id").value.trim();
  const videoId = document.getElementById("video_id").value.trim();
  const topN = document.getElementById("top_n").value.trim() || "10";

  if (!userId) {
    setError("Enter a User ID, username, or email.");
    return;
  }

  recommendBtn.disabled = true;
  setLoading("Generating recommendations...");

  try {
    const params = new URLSearchParams({ user_id: userId, top_n: topN });
    if (videoId) {
      params.set("video_id", videoId);
    }
    const data = await requestJson(`/recommend?${params.toString()}`);
    renderReelTable(`Recommendations for ${data.username || data.userId || userId}`, data);
  } catch (error) {
    setError(error.message);
  } finally {
    recommendBtn.disabled = false;
  }
}

async function getTrending() {
  trendingBtn.disabled = true;
  setLoading("Loading trending reels...");

  try {
    const topN = document.getElementById("top_n").value.trim() || "20";
    const data = await requestJson(`/trending?top_n=${encodeURIComponent(topN)}`);
    renderReelTable("Trending reels", data);
  } catch (error) {
    setError(error.message);
  } finally {
    trendingBtn.disabled = false;
  }
}

function renderHealthSection(title, status) {
  const collections = status.collections
    ? Object.entries(status.collections)
        .map(([name, count]) => `<tr><td>${name}</td><td>${count === null ? "Not available" : count.toLocaleString()}</td></tr>`)
        .join("")
    : "";

  return `
    <div class="panel">
      <h2>${title}: ${status.status}</h2>
      ${status.message ? `<p>${status.message}</p>` : ""}
      ${
        collections
          ? `<div class="table-wrap"><table><thead><tr><th>Collection</th><th>Documents</th></tr></thead><tbody>${collections}</tbody></table></div>`
          : ""
      }
    </div>
  `;
}

async function getHealth() {
  healthBtn.disabled = true;
  setLoading("Checking engine health...");

  try {
    const data = await requestJson("/health");
    resultsDiv.innerHTML = `
      <div class="panel">
        <h2>${data.message}</h2>
        <p>Overall status: ${data.status}</p>
      </div>
      ${renderHealthSection("Reel engine", data.reel_engine)}
      ${renderHealthSection("Product engine", data.product_engine)}
    `;
  } catch (error) {
    setError(error.message);
  } finally {
    healthBtn.disabled = false;
  }
}

async function getZatchRecommendations() {
  const userId = document.getElementById("zatch_user_id").value.trim();
  const reelId = document.getElementById("zatch_reel_id").value.trim();
  const limit = document.getElementById("zatch_limit").value.trim() || "10";

  if (!userId) {
    setError("Enter a Zatch User ID, username, or email.");
    return;
  }

  zatchRecommendBtn.disabled = true;
  setLoading("Generating live Zatch recommendations...");

  try {
    const params = new URLSearchParams({ limit });
    if (reelId) {
      params.set("current_reel_id", reelId);
    }
    const data = await requestJson(`/zatch/reel-recommendations/${encodeURIComponent(userId)}?${params.toString()}`);
    renderReelTable(`Live Zatch recommendations for ${data.username || data.userId}`, data);
  } catch (error) {
    setError(error.message);
  } finally {
    zatchRecommendBtn.disabled = false;
  }
}

async function getZatchHealth() {
  zatchHealthBtn.disabled = true;
  setLoading("Checking Zatch MongoDB connection...");

  try {
    const data = await requestJson("/zatch/health");
    resultsDiv.innerHTML = renderHealthSection("Zatch MongoDB", data.zatch_mongodb);
  } catch (error) {
    setError(error.message);
  } finally {
    zatchHealthBtn.disabled = false;
  }
}

document.addEventListener("DOMContentLoaded", () => {
  recommendBtn.addEventListener("click", getRecommendations);
  trendingBtn.addEventListener("click", getTrending);
  healthBtn.addEventListener("click", getHealth);
  zatchRecommendBtn.addEventListener("click", getZatchRecommendations);
  zatchHealthBtn.addEventListener("click", getZatchHealth);
  getHealth();
});
