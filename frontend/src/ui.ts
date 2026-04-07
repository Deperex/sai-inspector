import { loadBundle, type LoadedBundle } from "./loadBundle";
import { MujocoDebugViewer } from "./viewer";

function getManifestUrlFromLocation(): string {
  const params = new URLSearchParams(window.location.search);
  const env = params.get("env") ?? "libero-spatial-v0";
  const task = params.get("task") ?? "0";
  return `/exports/${env}/task_${task}/manifest.json`;
}

function makeButton(label: string): HTMLButtonElement {
  const btn = document.createElement("button");
  btn.textContent = label;
  return btn;
}

const root = document.getElementById("app");
if (!root) {
  throw new Error("#app not found");
}

root.innerHTML = `
  <div class="layout">
    <div class="sidebar">
      <h1>SAI Inspector</h1>

      <div class="block">
        <label for="manifestUrl">Manifest URL</label>
        <input id="manifestUrl" />
      </div>

      <div class="block">
        <label>Status</label>
        <div id="status" class="meta status-warn">Idle</div>
      </div>

      <div class="block">
        <label>Instruction</label>
        <div id="instruction" class="meta">-</div>
      </div>

      <div class="block">
        <label>Cameras</label>
        <select id="cameraSelect"></select>
      </div>

      <div class="block">
        <div id="buttonRow"></div>
      </div>

      <div class="block">
        <label>Manifest</label>
        <div id="meta" class="meta">-</div>
      </div>
    </div>

    <div class="viewer" id="viewer"></div>
  </div>
`;

const manifestInput = root.querySelector<HTMLInputElement>("#manifestUrl");
const statusEl = root.querySelector<HTMLDivElement>("#status");
const instructionEl = root.querySelector<HTMLDivElement>("#instruction");
const metaEl = root.querySelector<HTMLDivElement>("#meta");
const cameraSelect = root.querySelector<HTMLSelectElement>("#cameraSelect");
const viewerEl = root.querySelector<HTMLDivElement>("#viewer");
const buttonRow = root.querySelector<HTMLDivElement>("#buttonRow");

if (
  !manifestInput ||
  !statusEl ||
  !instructionEl ||
  !metaEl ||
  !cameraSelect ||
  !viewerEl ||
  !buttonRow
) {
  throw new Error("Failed to find required UI elements");
}

manifestInput.value = getManifestUrlFromLocation();

const loadBtn = makeButton("Load bundle");
const playBtn = makeButton("Play");
const pauseBtn = makeButton("Pause");
const resetBtn = makeButton("Reset");

buttonRow.append(loadBtn, playBtn, pauseBtn, resetBtn);

const viewer = new MujocoDebugViewer(viewerEl);

let loaded: LoadedBundle | null = null;
let running = false;
let raf = 0;

function setStatus(text: string, kind: "ok" | "warn" | "err" = "warn") {
  statusEl.textContent = text;
  statusEl.className = `meta status-${kind}`;
}

function fillManifestInfo() {
  if (!loaded) {
    instructionEl.textContent = "-";
    metaEl.textContent = "-";
    cameraSelect.innerHTML = "";
    return;
  }

  instructionEl.textContent = loaded.manifest.instruction;

  metaEl.textContent =
    `env: ${loaded.manifest.env_name}\n` +
    `benchmark: ${loaded.manifest.benchmark_name}\n` +
    `task_id: ${loaded.manifest.task_id}\n` +
    `assets: ${loaded.manifest.assets.length}`;

  cameraSelect.innerHTML = "";

  for (const camName of loaded.manifest.cameras) {
    const opt = document.createElement("option");
    opt.value = camName;
    opt.textContent = camName;
    cameraSelect.appendChild(opt);
  }
}

function applySelectedCamera() {
  if (!loaded) return;
  const name = cameraSelect.value;
  if (!name) return;
  viewer.setActiveCameraByName(name);
}

function stopLoop() {
  running = false;
  if (raf) {
    cancelAnimationFrame(raf);
    raf = 0;
  }
}

function frame() {
  if (!loaded) {
    return;
  }

  if (running) {
    loaded.mujoco.mj_step(loaded.model, loaded.data);
    viewer.syncFromPhysics();
  }

  viewer.render();
  raf = requestAnimationFrame(frame);
}

async function loadCurrentBundle() {
  stopLoop();
  setStatus("Loading bundle...", "warn");

  try {
    loaded = await loadBundle(manifestInput.value.trim());
    await viewer.setLoadedScene(loaded);
    resetState();
    fillManifestInfo();
    applySelectedCamera();
    setStatus("Bundle loaded into MuJoCo WASM", "ok");
    running = false;
    frame();
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(msg, "err");
    console.error(err);
  }
}

function resetState() {
  if (!loaded) {
    return;
  }

  loaded.mujoco.mj_resetData(loaded.model, loaded.data);

  if (Array.isArray(loaded.state.qpos)) {
    const n = Math.min(loaded.state.qpos.length, loaded.data.qpos.length);
    for (let i = 0; i < n; i++) {
      loaded.data.qpos[i] = loaded.state.qpos[i];
    }
  }

  if (Array.isArray(loaded.state.qvel)) {
    const n = Math.min(loaded.state.qvel.length, loaded.data.qvel.length);
    for (let i = 0; i < n; i++) {
      loaded.data.qvel[i] = loaded.state.qvel[i];
    }
  }

  loaded.mujoco.mj_forward(loaded.model, loaded.data);
  viewer.syncFromPhysics();
}

loadBtn.onclick = () => {
  void loadCurrentBundle();
};

playBtn.onclick = () => {
  if (!loaded) {
    return;
  }
  running = true;
  setStatus("Running", "ok");
};

pauseBtn.onclick = () => {
  running = false;
  setStatus("Paused", "warn");
};

resetBtn.onclick = () => {
  if (!loaded) {
    return;
  }
  resetState();
  applySelectedCamera();
  setStatus("Reset to exported state", "ok");
};

cameraSelect.onchange = () => {
  applySelectedCamera();
};
