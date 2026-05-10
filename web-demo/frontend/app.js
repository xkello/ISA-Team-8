const statusEl = document.getElementById("status");
const inputEl = document.getElementById("userIdInput");
const minReviewsEl = document.getElementById("minReviewsInput");
const minReviewsGroupEl = document.getElementById("minReviewsGroup");
const coldStartToggleEl = document.getElementById("coldStartToggle");
const coldStartGroupEl = document.getElementById("coldStartGroup");
const posRatioEl = document.getElementById("posRatioInput");
const posRatioValueEl = document.getElementById("posRatioValue");
const posRatioGroupEl = document.getElementById("posRatioGroup");
const avgRatingEl = document.getElementById("avgRatingInput");
const avgRatingValueEl = document.getElementById("avgRatingValue");
const avgRatingGroupEl = document.getElementById("avgRatingGroup");
const thirdReviewEl = document.getElementById("thirdReviewInput");
const thirdReviewGroupEl = document.getElementById("thirdReviewGroup");
const clearFiltersBtn = document.getElementById("clearFiltersBtn");
const randomBtn = document.getElementById("randomBtn");
const confirmBtn = document.getElementById("confirmBtn");
const userPanelEl = document.getElementById("userPanel");
const qualityPanelEl = document.getElementById("qualityPanel");
const modelCardHostEl = document.getElementById("modelCardHost");
const modelTitleEl = document.getElementById("modelTitle");
const modelPositionEl = document.getElementById("modelPosition");
const modelDotsEl = document.getElementById("modelDots");
const modelPrevBtn = document.getElementById("modelPrevBtn");
const modelNextBtn = document.getElementById("modelNextBtn");
const mapEl = document.getElementById("map");
const EUAIModal = document.getElementById("EUAIModal");
const EUAIModalCloseSpan = document.getElementById("EUAIModalCloseSpan");


// map centered on Philadelphia
const map = L.map('map').setView([39.9528, -75.1636], 13);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    maxZoom: 19,
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
}).addTo(map);
let allMapMarkers = [];

const MODEL_KEYS = ["lstm", "naive_bayes", "hybrid"];
const MODEL_TITLES = {
  lstm: "LSTM (ordered sequence)",
  naive_bayes: "Naive Bayes (unordered)",
  hybrid: "Hybrid",
};

let currentModelIndex = 0;
let currentResponse = null;

function setStatus(msg) {
  statusEl.textContent = msg;
}

function updateSliderLabels() {
  const ratio = Number(posRatioEl.value);
  posRatioValueEl.textContent = ratio === 0 ? "any" : `≥ ${ratio}%`;

  const rating = Number(avgRatingEl.value);
  avgRatingValueEl.textContent = rating <= 1 ? "any" : `≥ ${rating.toFixed(1)}`;
}

function hasActiveFilters() {
  const hasMinReviews = minReviewsEl.value.trim() !== "";
  const hasPosRatio = Number(posRatioEl.value) > 0;
  const hasAvgRating = Number(avgRatingEl.value) > 1;
  const hasDate = thirdReviewEl.value.trim() !== "";
  return hasMinReviews || hasPosRatio || hasAvgRating || hasDate;
}

function syncRandomFilters(source = null) {
  const coldStartEnabled = coldStartToggleEl.checked;

  // if cold start just turned on, clear all filter fields
  if (source === "cold-start" && coldStartEnabled) {
    minReviewsEl.value = "";
    posRatioEl.value = "0";
    avgRatingEl.value = "1";
    thirdReviewEl.value = "";
    updateSliderLabels();
  }

  // if any filter is now set, uncheck cold start
  if (source !== "cold-start" && hasActiveFilters() && coldStartToggleEl.checked) {
    coldStartToggleEl.checked = false;
  }

  const anyFilter = hasActiveFilters();
  const nextColdStart = coldStartToggleEl.checked;

  // disable/grey cold start when filters are set
  coldStartToggleEl.disabled = anyFilter;
  coldStartGroupEl?.classList.toggle("is-disabled", anyFilter);

  // disable/grey filter controls when cold start is active
  const filterEls = [minReviewsGroupEl, posRatioGroupEl, avgRatingGroupEl, thirdReviewGroupEl];
  filterEls.forEach(el => el?.classList.toggle("is-disabled", nextColdStart));
  minReviewsEl.disabled = nextColdStart;
  posRatioEl.disabled = nextColdStart;
  avgRatingEl.disabled = nextColdStart;
  thirdReviewEl.disabled = nextColdStart;

  updateSliderLabels();
}

function clearFilters() {
  minReviewsEl.value = "";
  posRatioEl.value = "0";
  avgRatingEl.value = "1";
  thirdReviewEl.value = "";
  coldStartToggleEl.checked = false;
  syncRandomFilters();
}

function fmtPct(v) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return `${(v * 100).toFixed(2)}%`;
}

function fmtNum(v, digits = 2) {
  if (v === null || v === undefined || Number.isNaN(v)) return "n/a";
  return Number(v).toFixed(digits);
}

function renderQualityPanel(modelKey = null, metric = {}) {
  const modelLabel = modelKey ? MODEL_TITLES[modelKey] : "Model not selected";
  qualityPanelEl.innerHTML = `
    <div class="quality-head">
      <h3>Quality Evaluation and Risk Assessment</h3>
      <button id="EUAIBtn" class="modal-open-btn">Open full assessment</button>
    </div>
    <div class="small quality-sub">Current focus: ${modelLabel}</div>
    <div class="quality-metrics">
      <span class="metric-pill">Top-10 success: ${fmtPct(metric.top10_success)}</span>
      <span class="metric-pill">Top-5 success: ${fmtPct(metric.top5_success)}</span>
    </div>
    <p class="quality-text">
      Summary: this is a domain-specific restaurant recommender, not a general-purpose AI system. Main risks are sparse-history users and possible recommendation bias; mitigation focuses on transparent outputs and user controls.
    </p>
    <p class="quality-text">
      Performance generally improves as user history grows. Rich interaction history gives collaborative and sequence models more reliable signal.
    </p>
    <p class="quality-text">
      Main risk areas are cold-start users and sparse-history users. In these cases, popularity and content-based fallbacks help, but ranking quality can remain limited.
    </p>
    <p class="quality-text">
      Mitigation options include adding richer content features and testing additional classifiers or ranking approaches for low-history segments.
    </p>
  `;
}

function renderUserPanel(userId, info) {
  if (!info || Object.keys(info).length === 0) {
    userPanelEl.hidden = false;
    userPanelEl.innerHTML = `
      <div class="user-panel-header">
        <h3>User Profile</h3>
        <span class="small">${userId}</span>
      </div>
      <div class="small">No extra user stats available for this user.</div>
    `;
    return;
  }

  const visits = Array.isArray(info.recent_visits) ? info.recent_visits : [];
  const topTypes = Array.isArray(info.top_restaurant_types) ? info.top_restaurant_types : [];
  userPanelEl.hidden = false;
  userPanelEl.innerHTML = `
    <div class="user-panel-header">
      <h3>User Profile</h3>
      <span class="small">${userId}</span>
    </div>
    ${info.profile_note ? `<div class="small profile-note">${info.profile_note}</div>` : ""}
    <div class="user-stats-grid">
      <div class="metric-pill">Avg rating: ${fmtNum(info.avg_rating, 2)}</div>
      <div class="metric-pill">Reviews: ${info.review_count ?? "n/a"}</div>
      <div class="metric-pill">Unique restaurants: ${info.unique_restaurants ?? "n/a"}</div>
      <div class="metric-pill">Positive ratio: ${fmtPct(info.positive_review_ratio)}</div>
    </div>
    <div class="small recent-title">Most popular restaurant types</div>
    <div class="type-chip-wrap">
      ${topTypes.length ? topTypes.map((t) => `<span class="type-chip">${t.type} (${t.count})</span>`).join("") : '<span class="small">No type data found.</span>'}
    </div>
    <div class="small recent-title">Last 3 visited restaurants</div>
    <div class="user-visit-list">
      ${visits.length ? visits.map((v) => `
        <div class="user-visit-row">
          <div>${v.name || "Unknown"}</div>
          <div>Rating: ${fmtNum(v.rating, 1)}</div>
          <div>${v.date || ""}</div>
        </div>
      `).join("") : '<div class="small">No visit history found.</div>'}
    </div>
  `;
}

function renderModel(modelKey, data) {
  const metric = data.metric || {};
  const recs = data.recommendations || [];

  const card = document.createElement("div");
  card.className = "model-card";
  card.innerHTML = `
    <div class="model-header">
      <div>
        <h3>${MODEL_TITLES[modelKey] || modelKey}</h3>
        <div class="small">${metric.notes || ""}</div>
      </div>
      <div class="model-header-metrics">
        <span class="metric-pill">Top-10 success: ${fmtPct(metric.top10_success)}</span>
        <span class="metric-pill">Top-5 success: ${fmtPct(metric.top5_success)}</span>
      </div>
    </div>
    <div class="reco-scroll">
      <div class="reco-row header">
        <div>Name</div>
        <div>Categories</div>
        <div>Rating</div>
        <div>Last review</div>
      </div>
      ${recs.map((r) => `
        <div class="reco-row">
          <div>${r.name || "Unknown"}</div>
          <div>${r.categories || ""}</div>
          <div>${(r.rating ?? 0).toFixed(1)}</div>
          <div>${r.last_review || "No review text available."}</div>
        </div>
      `).join("")}
    </div>
  `;

  // when recommendation div is clicked the restaurant pops-up on the map
  const recoScroll = card.querySelector('.reco-scroll');
  if (recoScroll) {
    recoScroll.addEventListener('click', (e) => {
      const row = e.target.closest('.reco-row');
      if (!row || row.classList.contains('header')) return;
      const cells = row.querySelectorAll('div');
      const name = (cells[0] && cells[0].textContent) || '';
      allMapMarkers.forEach(
        marker => {
          if (marker["name"] === name){
            marker["actualMarker"].openPopup();
          }
      });
    });
  }

  allMapMarkers.forEach(
    marker => {
      marker["actualMarker"].remove();
  });

  allMapMarkers = [];

  let latitudeSum = 0;
  let longitudeSum = 0;
  let plottedCount = 0;
  recs.forEach(rec => {
    const lat = Number(rec.latitude);
    const lon = Number(rec.longitude);
    if (!Number.isFinite(lat) || !Number.isFinite(lon)) {
      return;
    }
    latitudeSum += lat;
    longitudeSum += lon;
    plottedCount += 1;
    var newMarkerObject = {
      "actualMarker": L.marker([lat, lon]),
      "name": rec.name
    };
    newMarkerObject["actualMarker"].addTo(map)
    .bindPopup(`
      <b>${rec.name || "Unknown"} (${(rec.rating ?? 0).toFixed(1)}★)</b>
      <p>${rec.address || "Unknown address"}</p>
      <hr>
      <p>${rec.categories || ""}</p>
    `);

    allMapMarkers.push(newMarkerObject);
  });

  // center the map
  if (plottedCount > 0){
    map.setView([latitudeSum / plottedCount, longitudeSum / plottedCount], 13);
  }
  return card;
}

function renderModelDots() {
  if (!modelDotsEl) {
    return;
  }
  modelDotsEl.innerHTML = MODEL_KEYS.map((key, idx) => {
    const active = idx === currentModelIndex;
    const title = MODEL_TITLES[key] || key;
    return `<button class="dot-btn${active ? " active" : ""}" data-model-index="${idx}" aria-label="Show ${title}" title="${title}"></button>`;
  }).join("");
}

function renderActiveModel() {
  if (!currentResponse) {
    modelCardHostEl.innerHTML = "<div class='small'>No model results loaded yet.</div>";
    modelTitleEl.textContent = "Model gallery";
    modelPositionEl.textContent = "Select a user to load model outputs.";
    renderModelDots();
    renderQualityPanel();
    return;
  }

  const modelKey = MODEL_KEYS[currentModelIndex];
  const modelData = currentResponse.models?.[modelKey] || {};
  modelCardHostEl.innerHTML = "";
  modelCardHostEl.appendChild(renderModel(modelKey, modelData));
  modelTitleEl.textContent = MODEL_TITLES[modelKey] || modelKey;
  modelPositionEl.textContent = `Model ${currentModelIndex + 1} / ${MODEL_KEYS.length}`;
  renderModelDots();
  renderQualityPanel(modelKey, modelData.metric || {});
}

function shiftModel(step) {
  if (!currentResponse) {
    setStatus("Load recommendations first.");
    return;
  }
  currentModelIndex = (currentModelIndex + step + MODEL_KEYS.length) % MODEL_KEYS.length;
  renderActiveModel();
}

async function fetchRandomUser() {
  const useColdStart = coldStartToggleEl.checked;
  const rawMinReviews = minReviewsEl.value.trim();
  const posRatio = Number(posRatioEl.value);
  const avgRating = Number(avgRatingEl.value);
  const thirdDate = thirdReviewEl.value.trim();

  if (useColdStart) {
    const res = await fetch("/api/users/random?cold_start=true");
    setStatus("Picking random user...");
    if (!res.ok) { throw new Error(`Random user failed (${res.status}): ${await res.text()}`); }
    const body = await res.json();
    if (useColdStart && !body.cold_start) {
      throw new Error("Cold-start selection is not available from the current backend/artifacts.");
    }
    inputEl.value = body.user_id || "";
    setStatus(`Random cold-start user selected: ${inputEl.value}`);
    return;
  }

  const params = new URLSearchParams();
  if (rawMinReviews !== "") params.set("min_reviews", rawMinReviews);
  if (posRatio > 0)         params.set("min_positive_ratio", (posRatio / 100).toFixed(2));
  if (avgRating > 1)        params.set("min_avg_rating", avgRating.toFixed(1));
  if (thirdDate !== "")     params.set("max_third_review_date", thirdDate);

  const endpoint = `/api/users/random${params.toString() ? "?" + params.toString() : ""}`;
  setStatus("Picking random user...");
  const res = await fetch(endpoint);
  if (!res.ok) { throw new Error(`Random user failed (${res.status}): ${await res.text()}`); }
  const body = await res.json();
  inputEl.value = body.user_id || "";

  const parts = [];
  if (rawMinReviews !== "") parts.push(`min reviews: ${rawMinReviews}`);
  if (posRatio > 0)         parts.push(`pos ratio ≥ ${posRatio}%`);
  if (avgRating > 1)        parts.push(`avg rating ≥ ${avgRating.toFixed(1)}`);
  if (thirdDate !== "")     parts.push(`3rd review before ${thirdDate}`);
  setStatus(`Random user selected: ${inputEl.value}${parts.length ? " (" + parts.join(", ") + ")" : ""}`);
}

async function fetchRecommendations() {
  const userId = inputEl.value.trim();
  if (!userId) {
    setStatus("Enter a user_id first.");
    return;
  }

  setStatus(`Loading recommendations for ${userId}...`);
  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId }),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Recommendation failed (${res.status}): ${errText}`);
  }

  currentResponse = await res.json();
  currentModelIndex = 0;
  const info = currentResponse.user_info || {};
  const isColdStartUser = Boolean(info.is_cold_start) || Number(info.review_count ?? 0) <= 0;
  if (isColdStartUser) {
    userPanelEl.hidden = true;
    userPanelEl.innerHTML = "";
  } else {
    renderUserPanel(userId, info);
  }
  renderActiveModel();
  setStatus(`Showing recommendations for ${userId}`);
}

modelPrevBtn.addEventListener("click", () => shiftModel(-1));
modelNextBtn.addEventListener("click", () => shiftModel(1));
minReviewsEl.addEventListener("input", () => syncRandomFilters("min-reviews"));
coldStartToggleEl.addEventListener("change", () => syncRandomFilters("cold-start"));
posRatioEl.addEventListener("input", () => syncRandomFilters("pos-ratio"));
avgRatingEl.addEventListener("input", () => syncRandomFilters("avg-rating"));
thirdReviewEl.addEventListener("input", () => syncRandomFilters("third-review"));
clearFiltersBtn.addEventListener("click", clearFilters);
modelDotsEl.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  const idxRaw = target.getAttribute("data-model-index");
  if (idxRaw === null) {
    return;
  }
  if (!currentResponse) {
    setStatus("Load recommendations first.");
    return;
  }
  const idx = Number(idxRaw);
  if (!Number.isInteger(idx) || idx < 0 || idx >= MODEL_KEYS.length) {
    return;
  }
  currentModelIndex = idx;
  renderActiveModel();
});

window.addEventListener("keydown", (event) => {
  const activeTag = (document.activeElement?.tagName || "").toUpperCase();
  const isTypingTarget =
    activeTag === "INPUT" ||
    activeTag === "TEXTAREA" ||
    activeTag === "SELECT" ||
    document.activeElement?.isContentEditable;
  if (isTypingTarget) {
    return;
  }
  if (event.key === "ArrowLeft") {
    event.preventDefault();
    shiftModel(-1);
  } else if (event.key === "ArrowRight") {
    event.preventDefault();
    shiftModel(1);
  }
});

randomBtn.addEventListener("click", async () => {
  try {
    await fetchRandomUser();
  } catch (e) {
    setStatus(String(e));
  }
});

confirmBtn.addEventListener("click", async () => {
  try {
    await fetchRecommendations();
  } catch (e) {
    setStatus(String(e));
  }
});

EUAIModalCloseSpan.addEventListener("click", function() {
  EUAIModal.style.display = "none";
});

window.addEventListener("click", (event) => {
  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return;
  }
  if (target.id === "EUAIBtn") {
    EUAIModal.style.display = "block";
    return;
  }
  if (target === EUAIModal) {
    EUAIModal.style.display = "none";
  }
});

renderQualityPanel();
renderActiveModel();
syncRandomFilters();
setStatus("Ready.");


