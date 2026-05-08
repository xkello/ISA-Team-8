const statusEl = document.getElementById("status");
const inputEl = document.getElementById("userIdInput");
const minReviewsEl = document.getElementById("minReviewsInput");
const categoryEl = document.getElementById("categoryInput");
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

categoriesResponse = fetch("./possible_categories.json")
  .then(categoriesResponse => {
    if (!categoriesResponse.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return categoriesResponse.json(); 
  })
  .then(categories => {
    console.log("Categories Array:", categories);
    
    const select = document.getElementById('categoryInput');
    select.innerHTML = '<option value="">-- Select --</option>';

    categories.forEach(cat => {
      const option = document.createElement('option');
      option.value = cat;
      option.textContent = cat;
      select.appendChild(option);
    });
  })
  .catch(error => {
    console.error("Error fetching or parsing JSON:", error);
  });

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
    <h3>Quality Evaluation and Risk Assessment</h3>
    <div class="small quality-sub">Current focus: ${modelLabel}</div>
    <div class="quality-metrics">
      <span class="metric-pill">Top-10 success: ${fmtPct(metric.top10_success)}</span>
      <span class="metric-pill">Top-5 success: ${fmtPct(metric.top5_success)}</span>
    </div>
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
  const rawMinReviews = minReviewsEl.value.trim();
  const desiredCategory = categoryEl.value.trim();
  let endpoint = "/api/users/random";
  if (rawMinReviews !== "") {
    const minReviews = Number(rawMinReviews);
    if (!Number.isInteger(minReviews) || minReviews < 0) {
      throw new Error("Min reviews must be a whole number >= 0.");
    }
    endpoint = `/api/users/random?min_reviews=${encodeURIComponent(minReviews)}`;
  }


  setStatus("Picking random user...");
  const res = await fetch(endpoint);
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Random user failed (${res.status}): ${errText}`);
  }
  const body = await res.json();
  inputEl.value = body.user_id || "";
  if (rawMinReviews !== "") {
    //setStatus(`Random user selected: ${inputEl.value} (min reviews: ${rawMinReviews})`);
    setStatus(`Random user selected: ${inputEl.value} (min reviews: ${rawMinReviews}) (desired category: ${desiredCategory})`);
  } else {
    setStatus(`Random user selected: ${inputEl.value}`);
  }
}

async function fetchRecommendations() {
  const userId = inputEl.value.trim();
  const desiredCategory = categoryEl.value.trim();
  if (!userId) {
    setStatus("Enter a user_id first.");
    return;
  }

  setStatus(`Loading recommendations for ${userId}...`);
  const res = await fetch("/api/recommend", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      user_id: userId,
      desired_category: desiredCategory
    }),
  });
  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Recommendation failed (${res.status}): ${errText}`);
  }

  currentResponse = await res.json();
  currentModelIndex = 0;
  renderUserPanel(userId, currentResponse.user_info);
  renderActiveModel();
  setStatus(`Showing recommendations for ${userId}`);
}

modelPrevBtn.addEventListener("click", () => shiftModel(-1));
modelNextBtn.addEventListener("click", () => shiftModel(1));
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

renderQualityPanel();
renderActiveModel();
setStatus("Ready.");


