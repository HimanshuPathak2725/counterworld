const evidence = window.EVIDENCE;
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];

const metricLabels = {
  ensemble_return_variance: 'Ensemble return variance',
  heldout_return: 'Held-out return',
  one_step_error: 'One-step error',
  state_feature_baseline: 'State-feature baseline',
  csr: 'Candidate-Set Regret (CSR)',
};

function renderMetricTable(regime) {
  const data = evidence.power.regimes[regime];
  const body = $('#metric-table tbody');
  body.replaceChildren();
  for (const [key, label] of Object.entries(metricLabels)) {
    const score = key === 'csr' ? data.heldout_return.csr_mean_spearman : data[key].baseline_mean_spearman;
    const required = key === 'csr' ? '—' : data[key].required_n;
    const within = key === 'csr' ? null : data[key].within_cap;
    const row = document.createElement('tr');
    const name = document.createElement('td');
    const correlation = document.createElement('td');
    const requiredCell = document.createElement('td');
    const cap = document.createElement('td');
    name.textContent = label;
    correlation.textContent = score.toFixed(3);
    correlation.className = score >= 0 ? 'positive' : 'negative';
    requiredCell.textContent = String(required);
    cap.textContent = within === null ? 'N/A' : within ? 'Yes' : 'No';
    cap.className = within === null ? '' : within ? 'positive' : 'negative';
    row.append(name, correlation, requiredCell, cap);
    body.append(row);
  }
}

function renderProvenance() {
  const list = $('#source-hashes');
  for (const [name, source] of Object.entries(evidence.source_artifacts)) {
    const wrapper = document.createElement('div');
    const term = document.createElement('dt');
    const hash = document.createElement('dd');
    term.textContent = name;
    hash.textContent = source.sha256;
    wrapper.append(term, hash);
    list.append(wrapper);
  }
  $('#selection-rule').textContent = evidence.selection.rule;
  const facts = [
    ['Checkpoint', evidence.selection.checkpoint_id],
    ['Median collision p', evidence.selection.median_collision_probability.toFixed(4)],
    ['Start state', evidence.selection.state.map((x) => x.toFixed(2)).join(', ')],
    ['Goal', evidence.selection.goal.map((x) => x.toFixed(2)).join(', ')],
    ['Gap center', evidence.selection.gap_center.toFixed(3)],
    ['Gap width', evidence.selection.gap_width.toFixed(2)],
    ['Noise σp / σv', `${evidence.selection.sigma_p} / ${evidence.selection.sigma_v}`],
  ];
  const container = $('#selection-facts');
  for (const [label, value] of facts) {
    const item = document.createElement('div');
    item.className = 'selection-fact';
    const left = document.createElement('span');
    const right = document.createElement('span');
    left.textContent = label;
    right.textContent = value;
    item.append(left, right);
    container.append(item);
  }
  const real = evidence.trajectories.model_in_environment;
  $('#trajectory-raw').textContent = [
    `return: ${real.total_return}`,
    `collision: ${real.collision}`,
    `real environment steps: ${real.real_environment_steps}`,
    `model calls: ${real.model_calls}`,
    `states: ${JSON.stringify(real.states)}`,
  ].join('\n');
}

$$('[data-regime]').forEach((button) => {
  button.addEventListener('click', () => {
    const regime = button.dataset.regime;
    $$('[data-regime]').forEach((item) => {
      const active = item === button;
      item.classList.toggle('is-active', active);
      item.setAttribute('aria-pressed', String(active));
    });
    const image = $('#spearman-figure');
    image.src = `figures/spearman-${regime}.svg`;
    image.alt = `Grouped bars comparing five collision-risk scores in ${regime}`;
    renderMetricTable(regime);
  });
});

$('#copy-command').addEventListener('click', async () => {
  const command = $('#reproduction-command').textContent;
  try {
    await navigator.clipboard.writeText(command);
    $('#copy-command').textContent = 'Copied';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents($('#reproduction-command'));
    selection.removeAllRanges();
    selection.addRange(range);
    $('#copy-command').textContent = 'Select and copy';
  }
});
$('#print-report').addEventListener('click', () => window.print());

const videoDialog = $('#video-dialog');
const video = $('#explainer-video');
$('#play-video').addEventListener('click', () => {
  if (typeof videoDialog.showModal === 'function') videoDialog.showModal();
  else videoDialog.setAttribute('open', '');
});
$('#close-video').addEventListener('click', () => {
  video.pause();
  videoDialog.close();
});
videoDialog.addEventListener('click', (event) => {
  if (event.target === videoDialog) {
    video.pause();
    videoDialog.close();
  }
});

const canvas = $('#trajectory-canvas');
const context = canvas.getContext('2d');
canvas.tabIndex = 0;
canvas.setAttribute('role', 'img');
const records = evidence.trajectories;
const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let running = !reducedMotion;
let speed = 1;
let startTime = null;
let animationFrame = null;

const allStates = Object.values(records).flatMap((record) => record.states);
const xValues = allStates.map((state) => state[0]).concat([0, evidence.selection.goal[0]]);
const yValues = allStates.map((state) => state[1]).concat([evidence.selection.gap_center, evidence.selection.goal[1]]);
const bounds = {
  minX: Math.min(...xValues) - 1,
  maxX: Math.max(...xValues) + 1,
  minY: Math.min(...yValues) - 0.6,
  maxY: Math.max(...yValues) + 0.6,
};

function point(state) {
  const padding = 48;
  const x = padding + ((state[0] - bounds.minX) / (bounds.maxX - bounds.minX)) * (canvas.width - padding * 2);
  const y = padding + (1 - (state[1] - bounds.minY) / (bounds.maxY - bounds.minY)) * (canvas.height - padding * 2);
  return [x, y];
}

function linePath(states, color, progress = states.length - 1) {
  if (!states.length) return;
  context.beginPath();
  const lastWhole = Math.min(Math.floor(progress), states.length - 1);
  states.slice(0, lastWhole + 1).forEach((state, index) => {
    const [x, y] = point(state);
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  if (lastWhole < states.length - 1) {
    const next = point(states[lastWhole + 1]);
    context.lineTo(next[0], next[1]);
  }
  context.strokeStyle = color;
  context.lineWidth = 3;
  context.lineJoin = 'round';
  context.lineCap = 'round';
  context.stroke();
}

function agent(state, color, radius = 9) {
  const [x, y] = point(state);
  context.beginPath();
  context.arc(x, y, radius, 0, Math.PI * 2);
  context.fillStyle = color;
  context.fill();
  context.lineWidth = 2;
  context.strokeStyle = '#f9f2e5';
  context.stroke();
}

function text(x, y, value, size = 15, color = '#696d74', align = 'left') {
  context.fillStyle = color;
  context.font = `500 ${size}px 'DejaVu Sans', sans-serif`;
  context.textAlign = align;
  context.fillText(value, x, y);
}

function drawScene(progress) {
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  context.fillStyle = '#f9f2e5';
  context.fillRect(0, 0, width, height);
  context.strokeStyle = '#e2d7c7';
  context.lineWidth = 1;
  for (let index = 1; index < 9; index += 1) {
    const x = (width / 9) * index;
    context.beginPath(); context.moveTo(x, 45); context.lineTo(x, height - 36); context.stroke();
  }
  for (let index = 1; index < 6; index += 1) {
    const y = 45 + ((height - 80) / 6) * index;
    context.beginPath(); context.moveTo(35, y); context.lineTo(width - 35, y); context.stroke();
  }

  text(36, 30, 'TRUE ENVIRONMENT', 12, '#18213d');
  text(width - 36, 30, `CHECKPOINT ${evidence.selection.checkpoint_id}`, 10, '#696d74', 'right');
  const wallX = point([0, 0])[0];
  const gapTop = point([0, evidence.selection.gap_center + evidence.selection.gap_width / 2])[1];
  const gapBottom = point([0, evidence.selection.gap_center - evidence.selection.gap_width / 2])[1];
  context.strokeStyle = '#18213d';
  context.lineWidth = 8;
  context.beginPath(); context.moveTo(wallX, 45); context.lineTo(wallX, gapTop); context.stroke();
  context.beginPath(); context.moveTo(wallX, gapBottom); context.lineTo(wallX, height - 35); context.stroke();
  context.strokeStyle = '#d98d22';
  context.lineWidth = 4;
  context.beginPath(); context.moveTo(wallX, gapTop); context.lineTo(wallX, gapBottom); context.stroke();
  text(wallX + 12, (gapTop + gapBottom) / 2, 'gap', 12, '#d98d22');

  const goal = point(evidence.selection.goal);
  context.beginPath(); context.arc(goal[0], goal[1], 13, 0, Math.PI * 2);
  context.fillStyle = '#d98d22'; context.fill();
  text(goal[0] + 18, goal[1] + 4, 'goal', 12, '#696d74');
  const start = records.model_in_environment.states[0];
  agent(start, '#1aa3a1', 8);
  text(point(start)[0] + 14, point(start)[1] + 4, 'start', 12, '#696d74');

  linePath(records.model_in_imagination.states, '#c0652d', progress);
  linePath(records.model_in_environment.states, '#1aa3a1', progress);
  linePath(records.reference_in_environment.states, '#18213d', progress);

  const last = records.model_in_environment.states.length - 1;
  const whole = Math.min(Math.floor(progress), last);
  const fraction = Math.min(progress - whole, 1);
  const currentA = records.model_in_environment.states[whole];
  const currentB = records.model_in_environment.states[Math.min(whole + 1, last)];
  const current = currentA.map((value, index) => value + (currentB[index] - value) * fraction);
  agent(current, '#1aa3a1', 10);
  text(width - 36, height - 16, `step ${Math.min(whole + 1, last + 1)} / ${last + 1}`, 11, '#696d74', 'right');
}

function animate(timestamp) {
  if (!startTime) startTime = timestamp;
  const progress = Math.min((timestamp - startTime) / (9000 / speed), records.model_in_environment.states.length - 1);
  drawScene(progress);
  if (progress < records.model_in_environment.states.length - 1 && running) {
    animationFrame = requestAnimationFrame(animate);
  } else {
    running = false;
    $('#trajectory-play').textContent = 'Play';
  }
}

$('#trajectory-play').addEventListener('click', () => {
  running = !running;
  $('#trajectory-play').textContent = running ? 'Pause' : 'Play';
  if (running) {
    if (animationFrame) cancelAnimationFrame(animationFrame);
    startTime = null;
    animationFrame = requestAnimationFrame(animate);
  } else if (animationFrame) {
    cancelAnimationFrame(animationFrame);
    animationFrame = null;
  }
});
$('#trajectory-restart').addEventListener('click', () => {
  startTime = null;
  if (animationFrame) cancelAnimationFrame(animationFrame);
  running = true;
  $('#trajectory-play').textContent = 'Pause';
  animationFrame = requestAnimationFrame(animate);
});
$('#trajectory-speed').addEventListener('input', (event) => {
  speed = Number(event.target.value);
  $('#trajectory-speed-value').textContent = `${speed}×`;
});

renderMetricTable('R0');
renderProvenance();
drawScene(reducedMotion ? records.model_in_environment.states.length - 1 : 0);
if (!reducedMotion) animationFrame = requestAnimationFrame(animate);
else $('#trajectory-play').textContent = 'Play';
