import { loadBundle, type LoadedBundle } from "./loadBundle";
import { MujocoDebugViewer } from "./viewer";

type BackendState = {
  qpos: number[];
  qvel: number[];
};

type LiveScene = {
  env_name: string;
  benchmark_name: string;
  task_id: number;
  instruction: string;
  mutable_objects: string[];
  unknown_mutable_objects: string[];
  objects: Array<{
    name: string;
    kind: string;
    mobility: string;
    mutable_candidate: boolean;
    configured_mutable: boolean;
    joints: Array<{
      name: string;
      type: string;
      qpos_addr: number;
      dof_addr: number;
      axis?: number[];
      range?: number[];
    }>;
  }>;
};

type StartSessionResponse = {
  env_name: string;
  task_id: number;
  instruction: string;
  scene: LiveScene;
  state: BackendState;
};

type LiveSession = {
  envName: string;
  taskId: number;
  scene: LiveScene;
};

type ObjectStateResponse = {
  name: string;
  joints: Record<
    string,
    | {
        type: "free";
        pos: number[];
        quat: number[];
      }
    | {
        type: string;
        value: number;
      }
  >;
};

type ExportIndexEntry = {
  env_name: string;
  task_id: number;
  benchmark_name: string;
  instruction: string;
  manifest_url: string;
};

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
        <label for="exportSelect">Available Exports</label>
        <select id="exportSelect"></select>
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
        <label>Mode</label>
        <div id="modeRow" class="toggleRow"></div>
      </div>

      <div class="block">
        <label for="mutableObjectSelect">Setup Object</label>
        <select id="mutableObjectSelect"></select>
      </div>

      <div class="block">
        <label for="jointSelect">Setup Joint</label>
        <select id="jointSelect"></select>
      </div>

      <div class="block">
        <label>Setup Controls</label>
        <div id="setupControls" class="controlsPanel meta">Load an env to edit mutable objects.</div>
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
const exportSelect = root.querySelector<HTMLSelectElement>("#exportSelect");
const statusEl = root.querySelector<HTMLDivElement>("#status");
const instructionEl = root.querySelector<HTMLDivElement>("#instruction");
const metaEl = root.querySelector<HTMLDivElement>("#meta");
const cameraSelect = root.querySelector<HTMLSelectElement>("#cameraSelect");
const mutableObjectSelect = root.querySelector<HTMLSelectElement>("#mutableObjectSelect");
const jointSelect = root.querySelector<HTMLSelectElement>("#jointSelect");
const setupControls = root.querySelector<HTMLDivElement>("#setupControls");
const modeRow = root.querySelector<HTMLDivElement>("#modeRow");
const viewerEl = root.querySelector<HTMLDivElement>("#viewer");
const buttonRow = root.querySelector<HTMLDivElement>("#buttonRow");

if (
  !manifestInput ||
  !exportSelect ||
  !statusEl ||
  !instructionEl ||
  !metaEl ||
  !cameraSelect ||
  !mutableObjectSelect ||
  !jointSelect ||
  !setupControls ||
  !modeRow ||
  !viewerEl ||
  !buttonRow
) {
  throw new Error("Failed to find required UI elements");
}

const manifestInputEl = manifestInput;
const exportSelectEl = exportSelect;
const statusElEl = statusEl;
const instructionElEl = instructionEl;
const metaElEl = metaEl;
const cameraSelectEl = cameraSelect;
const mutableObjectSelectEl = mutableObjectSelect;
const jointSelectEl = jointSelect;
const setupControlsEl = setupControls;
const modeRowEl = modeRow;
const viewerElEl = viewerEl;
const buttonRowEl = buttonRow;

manifestInputEl.value = getManifestUrlFromLocation();

const loadBtn = makeButton("Load env");
const refreshBtn = makeButton("Refresh exports");
const playBtn = makeButton("Play");
const pauseBtn = makeButton("Pause");
const resetBtn = makeButton("Reset");
const inspectModeBtn = makeButton("Inspect");
const setupModeBtn = makeButton("Setup");

buttonRowEl.append(loadBtn, refreshBtn, playBtn, pauseBtn, resetBtn);
modeRowEl.append(inspectModeBtn, setupModeBtn);

const viewer = new MujocoDebugViewer(viewerElEl);

let loaded: LoadedBundle | null = null;
let liveSession: LiveSession | null = null;
let running = false;
let raf = 0;
let exportEntries: ExportIndexEntry[] = [];
let currentMode: "inspect" | "setup" = "inspect";
let selectedMutableObject = "";
let selectedJointName = "";
let objectStateCache: ObjectStateResponse | null = null;
const pressedKeys = new Set<string>();
let setupTickHandle = 0;
let setupMutationInFlight = false;

function setStatus(text: string, kind: "ok" | "warn" | "err" = "warn") {
  statusElEl.textContent = text;
  statusElEl.className = `meta status-${kind}`;
}

function fillManifestInfo() {
  if (!loaded) {
    instructionElEl.textContent = "-";
    metaElEl.textContent = "-";
    cameraSelectEl.innerHTML = "";
    mutableObjectSelectEl.innerHTML = "";
    jointSelectEl.innerHTML = "";
    setupControlsEl.textContent = "Load an env to edit mutable objects.";
    return;
  }

  instructionElEl.textContent = loaded.manifest.instruction;
  const liveText = liveSession
    ? `live: connected\nmutable objects: ${liveSession.scene.mutable_objects.length}`
    : "live: disconnected";

  metaElEl.textContent =
    `env: ${loaded.manifest.env_name}\n` +
    `benchmark: ${loaded.manifest.benchmark_name}\n` +
    `task_id: ${loaded.manifest.task_id}\n` +
    `assets: ${loaded.manifest.assets.length}\n` +
    liveText;

  cameraSelectEl.innerHTML = "";

  for (const camName of loaded.manifest.cameras) {
    const opt = document.createElement("option");
    opt.value = camName;
    opt.textContent = camName;
    cameraSelectEl.appendChild(opt);
  }

  fillMutableObjectOptions();
  fillJointOptions();
  renderSetupControls();
}

function applySelectedCamera() {
  if (!loaded) return;
  const name = cameraSelectEl.value;
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
  setStatus("Loading env...", "warn");

  try {
    const bundle = await loadBundle(manifestInputEl.value.trim());
    const live = await startLiveSession(bundle);
    loaded = bundle;
    liveSession = {
      envName: live.env_name,
      taskId: live.task_id,
      scene: live.scene,
    };
    await viewer.setLoadedScene(loaded);
    applySimState(live.state);
    fillManifestInfo();
    await loadSelectedObjectState();
    applySelectedCamera();
    setStatus("Env loaded and synced with live backend", "ok");
    running = false;
    frame();
  } catch (err) {
    liveSession = null;
    objectStateCache = null;
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(msg, "err");
    console.error(err);
  }
}

function fillMutableObjectOptions() {
  mutableObjectSelectEl.innerHTML = "";
  const objects = liveSession?.scene.objects.filter(
    (entry) => entry.mutable_candidate || entry.configured_mutable
  ) ?? [];

  for (const entry of objects) {
    const opt = document.createElement("option");
    opt.value = entry.name;
    opt.textContent = `${entry.name} (${entry.mobility})`;
    mutableObjectSelectEl.appendChild(opt);
  }

  if (objects.length === 0) {
    selectedMutableObject = "";
    return;
  }

  if (!objects.some((entry) => entry.name === selectedMutableObject)) {
    selectedMutableObject = objects[0].name;
  }
  mutableObjectSelectEl.value = selectedMutableObject;
}

function getSelectedSceneObject() {
  if (!liveSession || !selectedMutableObject) return null;
  return liveSession.scene.objects.find((entry) => entry.name === selectedMutableObject) ?? null;
}

function fillJointOptions() {
  jointSelectEl.innerHTML = "";
  const sceneObject = getSelectedSceneObject();
  if (!sceneObject || sceneObject.mobility !== "articulated") {
    selectedJointName = "";
    return;
  }

  for (const joint of sceneObject.joints) {
    const opt = document.createElement("option");
    opt.value = joint.name;
    opt.textContent = joint.name;
    jointSelectEl.appendChild(opt);
  }

  if (!sceneObject.joints.some((joint) => joint.name === selectedJointName)) {
    selectedJointName = sceneObject.joints[0]?.name ?? "";
  }
  jointSelectEl.value = selectedJointName;
}

function labelForExport(entry: ExportIndexEntry): string {
  return `${entry.env_name} / task_${entry.task_id}`;
}

function fillExportOptions() {
  exportSelectEl.innerHTML = "";
  for (const entry of exportEntries) {
    const opt = document.createElement("option");
    opt.value = entry.manifest_url;
    opt.textContent = `${labelForExport(entry)} - ${entry.instruction}`;
    exportSelectEl.appendChild(opt);
  }

  const matched = exportEntries.find((entry) => entry.manifest_url === manifestInputEl.value.trim());
  if (matched) {
    exportSelectEl.value = matched.manifest_url;
    return;
  }
  if (exportEntries.length > 0) {
    exportSelectEl.value = exportEntries[0].manifest_url;
    manifestInputEl.value = exportEntries[0].manifest_url;
  }
}

async function refreshExports() {
  const response = await fetch("/__exports_index");
  if (!response.ok) {
    throw new Error(`Failed to fetch export list: ${response.status}`);
  }
  const payload = (await response.json()) as { exports: ExportIndexEntry[] };
  exportEntries = payload.exports;
  fillExportOptions();
}

async function fetchObjectState(objectName: string): Promise<ObjectStateResponse> {
  const response = await fetch(`/api/mutable/object/${encodeURIComponent(objectName)}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to fetch object state: ${response.status} ${message}`);
  }
  return (await response.json()) as ObjectStateResponse;
}

async function mutateJoint(objectName: string, jointName: string, value: number): Promise<BackendState> {
  const response = await fetch("/api/mutable/joint", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      object: objectName,
      joint: jointName,
      value,
    }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to mutate joint: ${response.status} ${message}`);
  }
  const payload = (await response.json()) as { state: BackendState };
  return payload.state;
}

async function mutatePose(
  objectName: string,
  pos: number[],
  quat: number[],
): Promise<BackendState> {
  const response = await fetch("/api/mutable/pose", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      object: objectName,
      pos,
      quat,
    }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to mutate pose: ${response.status} ${message}`);
  }
  const payload = (await response.json()) as { state: BackendState };
  return payload.state;
}

async function loadSelectedObjectState() {
  if (!selectedMutableObject) {
    objectStateCache = null;
    renderSetupControls();
    return;
  }
  objectStateCache = await fetchObjectState(selectedMutableObject);
  fillJointOptions();
  renderSetupControls();
}

function applySimState(state: BackendState) {
  if (!loaded) {
    return;
  }

  loaded.mujoco.mj_resetData(loaded.model, loaded.data);

  if (Array.isArray(state.qpos)) {
    const n = Math.min(state.qpos.length, loaded.data.qpos.length);
    for (let i = 0; i < n; i++) {
      loaded.data.qpos[i] = state.qpos[i];
    }
  }

  if (Array.isArray(state.qvel)) {
    const n = Math.min(state.qvel.length, loaded.data.qvel.length);
    for (let i = 0; i < n; i++) {
      loaded.data.qvel[i] = state.qvel[i];
    }
  }

  loaded.mujoco.mj_forward(loaded.model, loaded.data);
  viewer.syncFromPhysics();
}

function resetState() {
  if (!loaded) return;
  applySimState(loaded.state as BackendState);
}

function setMode(mode: "inspect" | "setup") {
  currentMode = mode;
  inspectModeBtn.classList.toggle("is-active", mode === "inspect");
  setupModeBtn.classList.toggle("is-active", mode === "setup");
  if (mode !== "setup") {
    pressedKeys.clear();
  }
  renderSetupControls();
}

function renderSetupControls() {
  const sceneObject = getSelectedSceneObject();
  if (currentMode !== "setup") {
    setupControlsEl.textContent = "Switch to Setup mode to edit mutable objects.";
    return;
  }
  if (!sceneObject) {
    setupControlsEl.textContent = "No mutable object selected.";
    return;
  }

  setupControlsEl.innerHTML = "";

  if (sceneObject.mobility === "articulated") {
    const info = document.createElement("div");
    const activeJoint = sceneObject.joints.find((joint) => joint.name === selectedJointName) ?? sceneObject.joints[0];
    const cached = activeJoint ? objectStateCache?.joints[activeJoint.name] : null;
    const value = cached && "value" in cached ? cached.value : 0;
    const range = activeJoint?.range?.map((v) => v.toFixed(3)).join(" to ") ?? "unbounded";
    info.textContent =
      `Selected joint: ${activeJoint?.name ?? "-"}\n` +
      `Current value: ${value.toFixed(3)}\n` +
      `Range: ${range}\n\n` +
      `Keyboard:\n` +
      `ArrowLeft / [: decrease\n` +
      `ArrowRight / ]: increase\n` +
      `Tab: cycle joints`;
    setupControlsEl.appendChild(info);
    return;
  }

  if (sceneObject.mobility === "free") {
    const joint = sceneObject.joints[0];
    const cached = joint ? objectStateCache?.joints[joint.name] : null;
    if (!cached || !("pos" in cached)) {
      setupControlsEl.textContent = "Live pose unavailable for this object.";
      return;
    }
    const info = document.createElement("div");
    info.textContent =
      `Position:\n` +
      `x=${cached.pos[0].toFixed(3)} y=${cached.pos[1].toFixed(3)} z=${cached.pos[2].toFixed(3)}\n\n` +
      `Keyboard:\n` +
      `W/S: forward / back\n` +
      `A/D: left / right\n` +
      `R/F: up / down\n` +
      `Shift: faster movement`;
    setupControlsEl.appendChild(info);
    return;
  }

  setupControlsEl.textContent = "This object type is not editable yet.";
}

async function startLiveSession(bundle: LoadedBundle): Promise<StartSessionResponse> {
  const response = await fetch("/api/session/start", {
    method: "POST",
    headers: {
      "content-type": "application/json",
    },
    body: JSON.stringify({
      env: bundle.manifest.env_name,
      task_id: bundle.manifest.task_id,
    }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to start live session: ${response.status} ${message}`);
  }

  return (await response.json()) as StartSessionResponse;
}

async function resetLiveSession(): Promise<BackendState> {
  const response = await fetch("/api/session/reset", {
    method: "POST",
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to reset live session: ${response.status} ${message}`);
  }
  const payload = (await response.json()) as { state: BackendState };
  return payload.state;
}

function nudgeSelectedJoint(direction: -1 | 1) {
  const sceneObject = getSelectedSceneObject();
  if (!sceneObject || sceneObject.mobility !== "articulated" || !selectedJointName || setupMutationInFlight) {
    return;
  }
  const joint = sceneObject.joints.find((entry) => entry.name === selectedJointName);
  const cached = objectStateCache?.joints[selectedJointName];
  if (!joint || !cached || !("value" in cached)) return;

  const step = pressedKeys.has("shift") ? 0.05 : 0.01;
  let nextValue = cached.value + direction * step;
  if (joint.range) {
    nextValue = Math.min(Math.max(nextValue, joint.range[0]), joint.range[1]);
  }
  cached.value = nextValue;
  renderSetupControls();
  setupMutationInFlight = true;
  void (async () => {
    try {
      const state = await mutateJoint(sceneObject.name, selectedJointName, nextValue);
      applySimState(state);
      setStatus(`Updated ${selectedJointName}`, "ok");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
      await loadSelectedObjectState();
    } finally {
      setupMutationInFlight = false;
    }
  })();
}

function nudgeSelectedFreeObject() {
  const sceneObject = getSelectedSceneObject();
  if (!sceneObject || sceneObject.mobility !== "free" || setupMutationInFlight) {
    return;
  }
  const joint = sceneObject.joints[0];
  const cached = joint ? objectStateCache?.joints[joint.name] : null;
  if (!joint || !cached || !("pos" in cached)) return;

  const step = pressedKeys.has("shift") ? 0.02 : 0.005;
  const pos = [...cached.pos];
  if (pressedKeys.has("w")) pos[1] += step;
  if (pressedKeys.has("s")) pos[1] -= step;
  if (pressedKeys.has("a")) pos[0] -= step;
  if (pressedKeys.has("d")) pos[0] += step;
  if (pressedKeys.has("r")) pos[2] += step;
  if (pressedKeys.has("f")) pos[2] -= step;
  if (pos.every((value, index) => Math.abs(value - cached.pos[index]) < 1e-9)) {
    return;
  }

  cached.pos = pos;
  renderSetupControls();
  setupMutationInFlight = true;
  void (async () => {
    try {
      const state = await mutatePose(sceneObject.name, pos, cached.quat);
      applySimState(state);
      setStatus(`Moved ${sceneObject.name}`, "ok");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
      await loadSelectedObjectState();
    } finally {
      setupMutationInFlight = false;
    }
  })();
}

function setupTick() {
  if (currentMode !== "setup" || !liveSession || pressedKeys.size === 0) {
    return;
  }
  const sceneObject = getSelectedSceneObject();
  if (!sceneObject) return;
  if (sceneObject.mobility === "articulated") {
    if (pressedKeys.has("arrowleft") || pressedKeys.has("[")) {
      nudgeSelectedJoint(-1);
    } else if (pressedKeys.has("arrowright") || pressedKeys.has("]")) {
      nudgeSelectedJoint(1);
    }
    return;
  }
  if (sceneObject.mobility === "free") {
    nudgeSelectedFreeObject();
  }
}

loadBtn.onclick = () => {
  void loadCurrentBundle();
};

refreshBtn.onclick = () => {
  void (async () => {
    try {
      await refreshExports();
      setStatus(`Found ${exportEntries.length} exported envs`, "ok");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
    }
  })();
};

inspectModeBtn.onclick = () => {
  setMode("inspect");
};

setupModeBtn.onclick = () => {
  setMode("setup");
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
  void (async () => {
    try {
      if (liveSession) {
        const state = await resetLiveSession();
        applySimState(state);
        await loadSelectedObjectState();
        setStatus("Reset to live backend state", "ok");
      } else {
        resetState();
        setStatus("Reset to exported state", "ok");
      }
      applySelectedCamera();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
    }
  })();
};

cameraSelectEl.onchange = () => {
  applySelectedCamera();
};

exportSelectEl.onchange = () => {
  manifestInputEl.value = exportSelectEl.value;
};

mutableObjectSelectEl.onchange = () => {
  selectedMutableObject = mutableObjectSelectEl.value;
  void loadSelectedObjectState();
};

jointSelectEl.onchange = () => {
  selectedJointName = jointSelectEl.value;
  renderSetupControls();
};

window.addEventListener("keydown", (event) => {
  if (currentMode !== "setup") return;
  const tag = (event.target as HTMLElement | null)?.tagName?.toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;

  const key = event.key.toLowerCase();
  if (key === "tab") {
    const sceneObject = getSelectedSceneObject();
    if (sceneObject?.mobility === "articulated" && sceneObject.joints.length > 0) {
      event.preventDefault();
      const idx = sceneObject.joints.findIndex((joint) => joint.name === selectedJointName);
      const nextIdx = idx >= 0 ? (idx + 1) % sceneObject.joints.length : 0;
      selectedJointName = sceneObject.joints[nextIdx].name;
      jointSelectEl.value = selectedJointName;
      renderSetupControls();
    }
    return;
  }

  if (["w", "a", "s", "d", "r", "f", "shift", "arrowleft", "arrowright", "[", "]"].includes(key)) {
    event.preventDefault();
    pressedKeys.add(key);
  }
});

window.addEventListener("keyup", (event) => {
  const key = event.key.toLowerCase();
  pressedKeys.delete(key);
});

setupTickHandle = window.setInterval(setupTick, 80);

void (async () => {
  try {
    await refreshExports();
    setMode("inspect");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(msg, "err");
    console.error(err);
  }
})();
