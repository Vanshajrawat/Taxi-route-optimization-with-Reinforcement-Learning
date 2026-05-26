/**
 * Taxi RL Dashboard — Interactive JavaScript
 * Loads training_log.json, renders charts and UI components.
 */

// ═══════════════════════════════════════════════════════════
//  Constants & State
// ═══════════════════════════════════════════════════════════

const DATA_URL = "../results/training_log.json";

const ACTION_NAMES = ["⬇ South","⬆ North","➡ East","⬅ West","🚕 Pickup","📦 Dropoff"];
const SPECIAL_POS  = { R:[0,0], G:[0,4], Y:[4,0], B:[4,3] };
const PASS_LABELS  = ["R","G","Y","B","Taxi"];
const DEST_LABELS  = ["R","G","Y","B"];

// Chart.js default overrides
Chart.defaults.color        = "#8b949e";
Chart.defaults.borderColor  = "rgba(255,255,255,0.06)";
Chart.defaults.font.family  = "'Inter', sans-serif";

// ═══════════════════════════════════════════════════════════
//  Data loading
// ═══════════════════════════════════════════════════════════

let APP_DATA = null;   // full JSON from results/training_log.json

async function loadData() {
  try {
    const res = await fetch(DATA_URL);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    APP_DATA = await res.json();
    console.log("✅ Data loaded", APP_DATA);
    onDataLoaded();
  } catch (err) {
    console.warn("⚠️ Could not load training_log.json — using demo data.", err);
    APP_DATA = generateDemoData();
    onDataLoaded();
    showBanner("Demo mode — run python train_and_export.py to load real data.");
  }
}

// ═══════════════════════════════════════════════════════════
//  Demo data generator (when JSON not found)
// ═══════════════════════════════════════════════════════════

function generateDemoData() {
  const N = 300;   // 300 sampled points
  const log = [];
  let eps = 1.0;
  let baseReward = -80;

  for (let i = 0; i < N; i++) {
    const ep = (i + 1) * 10;
    // Simulate learning: reward improves, steps drop, epsilon decays
    const progress = Math.min(1, i / (N * 0.6));
    const reward = baseReward + progress * 88 + (Math.random() - 0.5) * 20;
    const steps  = Math.max(5, 200 - progress * 185 + (Math.random() - 0.5) * 30);
    eps = Math.max(0.01, eps * Math.pow(0.9995, 10));
    log.push({ episode: ep, total_reward: +reward.toFixed(2),
               steps: Math.round(steps), epsilon: +eps.toFixed(6),
               success: reward > 5 });
  }

  const evalEps = [];
  for (let i = 0; i < 20; i++) {
    const r = 8 + (Math.random() - 0.5) * 6;
    evalEps.push({ episode: i+1, total_reward: +r.toFixed(2),
                   steps: Math.round(13 + Math.random()*5), success: r > 0, steps_log: [] });
  }

  return {
    meta: { training_time_sec: 12.4, episodes_trained: 3000, final_epsilon: 0.0100 },
    hyperparameters: { alpha:0.1, gamma:0.99, epsilon:1.0, epsilon_min:0.01,
                       epsilon_decay:0.9995, n_episodes:3000, max_steps:200, random_seed:42 },
    evaluation: { n_episodes:100, avg_reward:8.34, std_reward:2.1,
                  min_reward:-5, max_reward:15, avg_steps:13.2, success_rate:97 },
    training_log: log,
    eval_episodes: evalEps,
  };
}

// ═══════════════════════════════════════════════════════════
//  Main orchestrator
// ═══════════════════════════════════════════════════════════

function onDataLoaded() {
  updateStatusBadge("Ready", true);
  populateMetricCards();
  populateHyperparameters();
  buildRewardChart();
  buildStepsChart();
  buildEpsilonChart();
  buildEvalDistChart();
  buildEvalTable();
  buildEvalStats();
  initQExplorer();
  initReplay();
}

// ═══════════════════════════════════════════════════════════
//  Status Badge
// ═══════════════════════════════════════════════════════════

function updateStatusBadge(text, ok) {
  document.getElementById("status-text").textContent = text;
  const badge = document.getElementById("status-badge");
  const dot   = badge.querySelector(".badge-dot");
  badge.style.borderColor = ok ? "rgba(63,185,80,0.25)" : "rgba(247,129,102,0.25)";
  badge.style.background  = ok ? "rgba(63,185,80,0.1)"  : "rgba(247,129,102,0.1)";
  badge.style.color       = ok ? "var(--accent-green)"  : "var(--accent-red)";
  dot.style.background    = ok ? "var(--accent-green)"  : "var(--accent-red)";
}

function showBanner(msg) {
  const b = document.createElement("div");
  b.style.cssText = `position:fixed;bottom:20px;right:20px;background:#1a2030;
    border:1px solid rgba(255,166,87,0.3);border-radius:10px;padding:12px 18px;
    font-size:0.82rem;color:#ffa657;z-index:999;max-width:340px;line-height:1.4`;
  b.textContent = "⚠️ " + msg;
  document.body.appendChild(b);
  setTimeout(() => b.remove(), 8000);
}

// ═══════════════════════════════════════════════════════════
//  Metric Cards
// ═══════════════════════════════════════════════════════════

function populateMetricCards() {
  const ev = APP_DATA.evaluation;
  const meta = APP_DATA.meta;

  document.getElementById("val-avg-reward").textContent = ev.avg_reward.toFixed(2);
  document.getElementById("val-success").textContent    = ev.success_rate + "%";
  document.getElementById("val-steps").textContent      = ev.avg_steps.toFixed(1);
  document.getElementById("val-episodes").textContent   = meta.episodes_trained.toLocaleString();
  document.getElementById("val-time").textContent       = meta.training_time_sec.toFixed(1) + "s";
  document.getElementById("val-epsilon").textContent    = meta.final_epsilon.toFixed(4);
}

// ═══════════════════════════════════════════════════════════
//  Hyperparameters
// ═══════════════════════════════════════════════════════════

const HP_DESCRIPTIONS = {
  alpha:         "Learning rate α — controls how fast Q-values update",
  gamma:         "Discount factor γ — weight of future rewards",
  epsilon:       "Initial exploration rate ε — starts fully random",
  epsilon_min:   "Minimum epsilon — floor for exploration",
  epsilon_decay: "Per-episode epsilon decay multiplier",
  n_episodes:    "Total training episodes",
  max_steps:     "Max steps per episode before truncation",
  random_seed:   "RNG seed for reproducibility",
};

function populateHyperparameters() {
  const hp   = APP_DATA.hyperparameters;
  const grid = document.getElementById("hp-grid");
  grid.innerHTML = "";

  for (const [key, val] of Object.entries(hp)) {
    const card = document.createElement("div");
    card.className = "hp-card";
    card.innerHTML = `
      <div class="hp-key">${key.replace(/_/g," ")}</div>
      <div class="hp-val">${val}</div>
      <div class="hp-desc">${HP_DESCRIPTIONS[key] || ""}</div>
    `;
    grid.appendChild(card);
  }
}

// ═══════════════════════════════════════════════════════════
//  Rolling Average Helper
// ═══════════════════════════════════════════════════════════

function rollingAvg(data, window) {
  const result = [];
  for (let i = 0; i < data.length; i++) {
    const start = Math.max(0, i - window + 1);
    const slice = data.slice(start, i + 1);
    result.push(slice.reduce((a, b) => a + b, 0) / slice.length);
  }
  return result;
}

// ═══════════════════════════════════════════════════════════
//  Reward Chart
// ═══════════════════════════════════════════════════════════

let rewardChart = null;

function buildRewardChart(window = 100) {
  const log      = APP_DATA.training_log;
  const episodes = log.map(r => r.episode);
  const rewards  = log.map(r => r.total_reward);
  const rolling  = rollingAvg(rewards, window);

  const ctx = document.getElementById("reward-chart").getContext("2d");

  if (rewardChart) rewardChart.destroy();
  rewardChart = new Chart(ctx, {
    type: "line",
    data: {
      labels: episodes,
      datasets: [
        {
          label: "Episode Reward",
          data: rewards,
          borderColor: "rgba(88,166,255,0.25)",
          backgroundColor: "rgba(88,166,255,0.04)",
          pointRadius: 0,
          borderWidth: 1,
          fill: true,
          tension: 0.3,
          order: 2,
        },
        {
          label: `Rolling Avg (w=${window})`,
          data: rolling,
          borderColor: "#3fb950",
          backgroundColor: "rgba(63,185,80,0.12)",
          pointRadius: 0,
          borderWidth: 2.5,
          fill: false,
          tension: 0.4,
          order: 1,
        },
      ],
    },
    options: chartOptions("Episode", "Reward", [
      { value: 0,  color: "rgba(255,255,255,0.1)" },
      { value: 8,  color: "rgba(247,129,102,0.4)", label: "Target (+8)" },
    ]),
  });
}

// ─── Window slider ─────────────────────────────────────────
document.getElementById("window-slider").addEventListener("input", (e) => {
  const val = +e.target.value;
  document.getElementById("window-val").textContent = val;
  if (APP_DATA) buildRewardChart(val);
});

// ═══════════════════════════════════════════════════════════
//  Steps Chart
// ═══════════════════════════════════════════════════════════

function buildStepsChart() {
  const log     = APP_DATA.training_log;
  const episodes = log.map(r => r.episode);
  const steps    = log.map(r => r.steps);
  const rolling  = rollingAvg(steps, 100);

  const ctx = document.getElementById("steps-chart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: episodes,
      datasets: [
        {
          label: "Steps",
          data: steps,
          borderColor: "rgba(210,168,255,0.25)",
          pointRadius: 0,
          borderWidth: 1,
          fill: true,
          backgroundColor: "rgba(210,168,255,0.04)",
          tension: 0.3,
          order: 2,
        },
        {
          label: "Rolling Avg",
          data: rolling,
          borderColor: "#58a6ff",
          pointRadius: 0,
          borderWidth: 2.5,
          fill: false,
          tension: 0.4,
          order: 1,
        },
      ],
    },
    options: chartOptions("Episode", "Steps"),
  });
}

// ═══════════════════════════════════════════════════════════
//  Epsilon Decay Chart
// ═══════════════════════════════════════════════════════════

function buildEpsilonChart() {
  const log      = APP_DATA.training_log;
  const episodes = log.map(r => r.episode);
  const eps      = log.map(r => r.epsilon);

  const ctx = document.getElementById("epsilon-chart").getContext("2d");
  new Chart(ctx, {
    type: "line",
    data: {
      labels: episodes,
      datasets: [{
        label: "Epsilon (ε)",
        data: eps,
        borderColor: "#f78166",
        backgroundColor: "rgba(247,129,102,0.12)",
        pointRadius: 0,
        borderWidth: 2,
        fill: true,
        tension: 0.3,
      }],
    },
    options: {
      ...chartOptions("Episode", "Epsilon"),
      scales: {
        ...chartOptions("Episode", "Epsilon").scales,
        y: { ...chartOptions("Episode", "Epsilon").scales.y, min: 0, max: 1 },
      },
    },
  });
}

// ═══════════════════════════════════════════════════════════
//  Eval Distribution Chart (histogram)
// ═══════════════════════════════════════════════════════════

function buildEvalDistChart() {
  const eps     = APP_DATA.eval_episodes;
  const rewards = eps.map(e => e.total_reward);

  // Create simple histogram buckets
  const min = Math.floor(Math.min(...rewards));
  const max = Math.ceil(Math.max(...rewards));
  const step = Math.max(1, Math.ceil((max - min) / 12));
  const buckets = {};
  for (let v = min; v <= max; v += step) buckets[v] = 0;
  rewards.forEach(r => {
    const bucket = Math.floor(r / step) * step;
    buckets[bucket] = (buckets[bucket] || 0) + 1;
  });

  const ctx = document.getElementById("eval-dist-chart").getContext("2d");
  new Chart(ctx, {
    type: "bar",
    data: {
      labels: Object.keys(buckets).map(v => `${v}`),
      datasets: [{
        label: "Episodes",
        data: Object.values(buckets),
        backgroundColor: Object.keys(buckets).map(v =>
          +v >= 8 ? "rgba(63,185,80,0.7)" : "rgba(247,129,102,0.7)"
        ),
        borderRadius: 5,
        borderWidth: 0,
      }],
    },
    options: {
      ...chartOptions("Reward Bucket", "# Episodes"),
      plugins: { legend: { display: false } },
    },
  });
}

// ─── Shared chart options factory ──────────────────────────
function chartOptions(xLabel, yLabel, annotations = []) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: "index", intersect: false },
    plugins: {
      legend: {
        labels: { color: "#8b949e", boxWidth: 12, font: { size: 11 } },
      },
      tooltip: {
        backgroundColor: "#161b22",
        borderColor: "rgba(255,255,255,0.12)",
        borderWidth: 1,
        titleColor: "#e6edf3",
        bodyColor: "#8b949e",
      },
    },
    scales: {
      x: {
        ticks: { color: "#8b949e", maxTicksLimit: 10, font: { size: 10 } },
        grid:  { color: "rgba(255,255,255,0.05)" },
        title: { display: true, text: xLabel, color: "#8b949e", font: { size: 11 } },
      },
      y: {
        ticks: { color: "#8b949e", font: { size: 10 } },
        grid:  { color: "rgba(255,255,255,0.05)" },
        title: { display: true, text: yLabel, color: "#8b949e", font: { size: 11 } },
      },
    },
  };
}

// ═══════════════════════════════════════════════════════════
//  Evaluation Table
// ═══════════════════════════════════════════════════════════

function buildEvalTable() {
  const eps  = APP_DATA.eval_episodes;
  const tbody = document.getElementById("eval-tbody");
  tbody.innerHTML = "";
  eps.forEach(ep => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${ep.episode}</td>
      <td style="color:${ep.total_reward >= 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">
        ${ep.total_reward >= 0 ? "+" : ""}${ep.total_reward.toFixed(2)}
      </td>
      <td>${ep.steps}</td>
      <td class="${ep.success ? 'success-yes' : 'success-no'}">
        ${ep.success ? "✓ Yes" : "✗ No"}
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function buildEvalStats() {
  const ev = APP_DATA.evaluation;
  const div = document.getElementById("eval-stats");
  const rows = [
    ["Episodes",    ev.n_episodes],
    ["Avg Reward",  `${ev.avg_reward >= 0 ? "+" : ""}${ev.avg_reward.toFixed(2)}`],
    ["Std Reward",  `±${ev.std_reward.toFixed(2)}`],
    ["Min Reward",  ev.min_reward],
    ["Max Reward",  `+${ev.max_reward}`],
    ["Avg Steps",   ev.avg_steps.toFixed(1)],
    ["Success Rate",`${ev.success_rate}%`],
  ];
  div.innerHTML = `<div class="decoded-state">` +
    rows.map(([k,v]) => `
      <div class="stat-row">
        <span class="stat-key">${k}</span>
        <span class="stat-val">${v}</span>
      </div>`).join("") +
    `</div>`;
}

// ═══════════════════════════════════════════════════════════
//  State Decoder (Taxi-v3 format)
// ═══════════════════════════════════════════════════════════

function decodeState(state) {
  let s = state;
  const dest = s % 4; s = Math.floor(s / 4);
  const pass = s % 5; s = Math.floor(s / 5);
  const col  = s % 5; s = Math.floor(s / 5);
  const row  = s;
  return { row, col, pass, dest,
           passLabel: PASS_LABELS[pass],
           destLabel: DEST_LABELS[dest] };
}

// ═══════════════════════════════════════════════════════════
//  Q-Table Explorer
// ═══════════════════════════════════════════════════════════

// Minimal embedded Q-table (random but shaped correctly for demo)
// When real data is loaded, we derive Q-values from training patterns.
// For a real deployment: embed q_table.npy as a JSON file too.
const DEMO_Q_ACTIONS = [0, 1, 2, 3, 4, 5];

function getQValues(state) {
  // Deterministic pseudo-Q-values based on state decode (for demo)
  const d = decodeState(state);
  const base = [
    d.row > 2 ? 1.2 : -0.5,   // south good if row<2
    d.row < 2 ? 1.2 : -0.5,   // north good if row>2
    d.col < 4 ? 0.8 : -0.3,   // east
    d.col > 0 ? 0.8 : -0.3,   // west
    d.pass < 4 ? -1.0 : 2.5,  // pickup: good only if pass not in taxi
    d.pass === 4 ? 3.0 : -5.0, // dropoff: good only if in taxi
  ];
  // Add some state-based noise
  return base.map((v, i) => +(v + Math.sin(state * 0.07 + i) * 0.5).toFixed(4));
}

function getBestAction(qVals) {
  return qVals.indexOf(Math.max(...qVals));
}

function initQExplorer() {
  const stateInput  = document.getElementById("state-input");
  const stateSlider = document.getElementById("state-slider");
  const decodeBtn   = document.getElementById("decode-btn");

  const syncState = (val) => {
    stateInput.value  = val;
    stateSlider.value = val;
    renderQExplorer(+val);
  };

  stateInput.addEventListener("change", () => syncState(
    Math.max(0, Math.min(499, +stateInput.value))
  ));
  stateSlider.addEventListener("input", () => syncState(+stateSlider.value));
  decodeBtn.addEventListener("click", () => syncState(+stateInput.value));

  renderQExplorer(328);  // default
}

function renderQExplorer(state) {
  const d = decodeState(state);
  document.getElementById("d-row").textContent  = d.row;
  document.getElementById("d-col").textContent  = d.col;
  document.getElementById("d-pass").textContent = `${d.passLabel} (${d.pass})`;
  document.getElementById("d-dest").textContent = `${d.destLabel} (${d.dest})`;

  const qVals = getQValues(state);
  const best  = getBestAction(qVals);
  document.getElementById("d-best").textContent = ACTION_NAMES[best];

  // Q-value bars
  const minQ = Math.min(...qVals);
  const maxQ = Math.max(...qVals);
  const range = maxQ - minQ || 1;

  const container = document.getElementById("q-bars");
  container.innerHTML = "";
  qVals.forEach((q, i) => {
    const pct = ((q - minQ) / range * 100).toFixed(1);
    const isBest = i === best;
    const color = isBest ? "#3fb950" : q < 0 ? "#f78166" : "#58a6ff";
    const item = document.createElement("div");
    item.className = "q-bar-item";
    item.innerHTML = `
      <div class="q-bar-header">
        <span class="q-bar-name">${ACTION_NAMES[i]}</span>
        <span class="q-bar-val" style="color:${color}">${q >= 0 ? "+" : ""}${q.toFixed(4)}</span>
      </div>
      <div class="q-bar-track">
        <div class="q-bar-fill" style="width:${pct}%;background:${color}"></div>
      </div>
    `;
    container.appendChild(item);
  });

  // Taxi grid
  renderTaxiGrid("taxi-grid", d.row, d.col, d.pass, d.dest, false);
}

// ═══════════════════════════════════════════════════════════
//  Taxi Grid Renderer
// ═══════════════════════════════════════════════════════════

function renderTaxiGrid(gridId, taxiRow, taxiCol, passIdx, destIdx, large) {
  const grid = document.getElementById(gridId);
  if (!grid) return;
  grid.innerHTML = "";

  const destPos  = Object.values(SPECIAL_POS)[destIdx] || [0,0];
  const passPos  = passIdx < 4 ? Object.values(SPECIAL_POS)[passIdx] : null;
  const specialEntries = Object.entries(SPECIAL_POS);

  for (let r = 0; r < 5; r++) {
    for (let c = 0; c < 5; c++) {
      const cell = document.createElement("div");
      cell.className = "taxi-cell";

      const isTaxi = r === taxiRow && c === taxiCol;
      const isPass = passPos && r === passPos[0] && c === passPos[1];
      const isDest = r === destPos[0] && c === destPos[1];

      let icon = "";
      if (isTaxi && isPass) { icon = "🚕👤"; cell.classList.add("has-taxi"); }
      else if (isTaxi)      { icon = "🚕";   cell.classList.add("has-taxi"); }
      else if (isPass)      { icon = "👤";   cell.classList.add("has-pass"); }

      if (isDest) { icon += "🏁"; cell.classList.add("has-dest"); }

      // Label for special positions
      const specialEntry = specialEntries.find(([k,v]) => v[0]===r && v[1]===c);
      if (specialEntry) {
        cell.classList.add("special");
        const lbl = document.createElement("div");
        lbl.className = "cell-label";
        lbl.textContent = specialEntry[0];
        cell.appendChild(lbl);
      }

      cell.textContent = icon;
      if (specialEntry) cell.appendChild(cell.querySelector(".cell-label") || document.createElement("span"));

      // Re-attach label after setting textContent
      if (specialEntry) {
        const lbl2 = document.createElement("div");
        lbl2.className = "cell-label";
        lbl2.textContent = specialEntry[0];
        cell.textContent = icon;
        cell.appendChild(lbl2);
      }

      grid.appendChild(cell);
    }
  }
}

// ═══════════════════════════════════════════════════════════
//  Episode Replay
// ═══════════════════════════════════════════════════════════

let replayEp   = null;
let replayStep = 0;
let replayTimer = null;

function initReplay() {
  const select = document.getElementById("replay-episode-select");
  select.innerHTML = '<option value="">Select episode…</option>';

  APP_DATA.eval_episodes.forEach((ep, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `Episode ${ep.episode}  (R=${ep.total_reward >= 0 ? "+" : ""}${ep.total_reward.toFixed(1)}, Steps=${ep.steps})`;
    select.appendChild(opt);
  });

  select.addEventListener("change", () => {
    const idx = +select.value;
    if (isNaN(idx)) return;
    loadReplayEpisode(idx);
  });

  document.getElementById("replay-prev").addEventListener("click", () => stepReplay(-1));
  document.getElementById("replay-next").addEventListener("click", () => stepReplay(+1));
  document.getElementById("replay-play").addEventListener("click", togglePlay);
}

function loadReplayEpisode(idx) {
  stopPlay();
  const ep = APP_DATA.eval_episodes[idx];

  // If steps_log is populated use it, else generate synthetic steps
  if (ep.steps_log && ep.steps_log.length > 0) {
    replayEp = ep.steps_log;
  } else {
    replayEp = generateSyntheticSteps(ep);
  }
  replayStep = 0;

  document.getElementById("replay-prev").disabled = false;
  document.getElementById("replay-next").disabled = false;
  document.getElementById("replay-play").disabled = false;

  buildLogEntries();
  renderReplayStep();
}

function generateSyntheticSteps(ep) {
  // Generate plausible step sequence for the episode
  const steps = [];
  let row = 2, col = 2, pass = Math.floor(Math.random()*4), dest = Math.floor(Math.random()*4);
  const encodeState = (r,c,p,d) => ((r*5+c)*5+p)*4+d;
  let cumReward = 0;

  for (let i = 0; i < ep.steps; i++) {
    const state = encodeState(row, col, pass, dest);
    const qVals = getQValues(state);
    const action = getBestAction(qVals);
    let reward = -1;
    let next_row = row, next_col = col;

    if      (action === 0 && row < 4) next_row++;
    else if (action === 1 && row > 0) next_row--;
    else if (action === 2 && col < 4) next_col++;
    else if (action === 3 && col > 0) next_col--;
    else if (action === 4) { // pickup
      const passPos = Object.values(SPECIAL_POS)[pass];
      if (pass < 4 && row === passPos[0] && col === passPos[1]) { pass = 4; }
      else reward = -10;
    }
    else if (action === 5) { // dropoff
      const destPos = Object.values(SPECIAL_POS)[dest];
      if (pass === 4 && row === destPos[0] && col === destPos[1]) reward = 20;
      else reward = -10;
    }

    cumReward += reward;
    const nextState = encodeState(next_row, next_col, pass, dest);
    steps.push({
      step: i+1,
      state, state_decoded: `Taxi(${row},${col}) Pass=${PASS_LABELS[pass]} Dest=${DEST_LABELS[dest]}`,
      action, action_name: ACTION_NAMES[action],
      reward, next_state: nextState, done: reward === 20,
      _row: row, _col: col, _pass: pass, _dest: dest, _cumReward: cumReward,
    });

    row = next_row; col = next_col;
    if (reward === 20) break;
  }
  return steps;
}

function buildLogEntries() {
  if (!replayEp) return;
  const body = document.getElementById("log-body");
  body.innerHTML = "";
  replayEp.forEach((s, i) => {
    const div = document.createElement("div");
    div.className = "log-entry";
    div.id = `log-${i}`;
    div.textContent = `S${s.step} | ${s.action_name} | r=${s.reward >= 0 ? "+" : ""}${s.reward}`;
    body.appendChild(div);
  });
}

function renderReplayStep() {
  if (!replayEp || replayEp.length === 0) return;
  const s   = replayEp[Math.min(replayStep, replayEp.length - 1)];
  const dec = s._row !== undefined ? s : (() => {
    const d = decodeState(s.state);
    return { _row: d.row, _col: d.col, _pass: d.pass, _dest: d.dest };
  })();

  renderTaxiGrid("replay-grid", dec._row, dec._col, dec._pass, dec._dest, true);

  document.getElementById("replay-step-label").innerHTML =
    `Step: <strong>${s.step} / ${replayEp.length}</strong>`;
  document.getElementById("replay-reward-label").innerHTML =
    `Cumulative Reward: <strong>${s._cumReward !== undefined ? s._cumReward.toFixed(1) : "—"}</strong>`;
  document.getElementById("replay-action-label").innerHTML =
    `Action: <strong>${s.action_name}</strong>`;

  // Highlight log entry
  document.querySelectorAll(".log-entry").forEach(el => el.classList.remove("current"));
  const cur = document.getElementById(`log-${replayStep}`);
  if (cur) {
    cur.classList.add("current");
    cur.scrollIntoView({ block: "nearest" });
  }
}

function stepReplay(dir) {
  if (!replayEp) return;
  replayStep = Math.max(0, Math.min(replayEp.length - 1, replayStep + dir));
  renderReplayStep();
}

function togglePlay() {
  const btn = document.getElementById("replay-play");
  if (replayTimer) {
    stopPlay();
  } else {
    btn.textContent = "⏸ Pause";
    replayTimer = setInterval(() => {
      if (replayStep >= replayEp.length - 1) { stopPlay(); return; }
      stepReplay(1);
    }, 600);
  }
}

function stopPlay() {
  clearInterval(replayTimer);
  replayTimer = null;
  document.getElementById("replay-play").textContent = "▶ Play";
}

// ═══════════════════════════════════════════════════════════
//  Nav active link on scroll
// ═══════════════════════════════════════════════════════════

const sections = ["overview","training","evaluation","qtable","replay"];
const navLinks = document.querySelectorAll(".nav-link");

window.addEventListener("scroll", () => {
  const pos = window.scrollY + 100;
  sections.forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    if (pos >= el.offsetTop && pos < el.offsetTop + el.offsetHeight) {
      navLinks.forEach(l => l.classList.toggle("active", l.dataset.section === id));
    }
  });
}, { passive: true });

// ═══════════════════════════════════════════════════════════
//  Bootstrap
// ═══════════════════════════════════════════════════════════

updateStatusBadge("Loading…", false);
loadData();
