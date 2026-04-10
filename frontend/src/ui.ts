import { loadBundle, type LoadedBundle } from "./loadBundle";
import { MujocoDebugViewer } from "./viewer";

type BackendState = {
  qpos: number[];
  qvel: number[];
};

type QposState = {
  qpos: number[];
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
  policy?: PolicyConfigResponse | null;
};

type LiveSession = {
  envName: string;
  taskId: number;
  scene: LiveScene;
  instruction: string;
  policy?: PolicyConfigResponse | null;
};

type PolicyModelEntry = {
  id: string;
  label: string;
  model_id: string;
  processor_id?: string | null;
  lora_id?: string | null;
  benchmark?: string;
};

type PolicyBackendEntry = {
  label: string;
  models: PolicyModelEntry[];
};

type PolicyCatalogResponse = {
  policies: Record<string, PolicyBackendEntry>;
};

type PolicyConfigResponse = {
  backend: string;
  model_id?: string | null;
  host: string;
  port: number;
  timeout_s: number;
  restart_if_needed?: boolean;
  resolved_model?: PolicyModelEntry;
};

type PolicyServerStatus = {
  running: boolean;
  healthy?: boolean;
  backend?: string;
  model_id?: string;
  host?: string;
  port?: number;
  pid?: number;
  log_path?: string;
  command?: string[];
};

type RolloutStepResponse = {
  instruction: string;
  steps: Array<{
    action: number[];
    reward: number;
    done: boolean;
  }>;
  state: BackendState;
  done: boolean;
};

type RolloutStreamEvent =
  | {
      type: "step";
      index: number;
      instruction: string;
      step: {
        reward: number;
        done: boolean;
      };
      state: QposState;
      done: boolean;
    }
  | {
      type: "done";
      instruction: string;
      steps_sent: number;
      done: boolean;
      state: QposState;
    }
  | {
      type: "error";
      error: string;
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
        <label>Rollout Backend</label>
        <select id="policyBackendSelect"></select>
      </div>

      <div class="block">
        <label>Rollout Model</label>
        <select id="policyModelSelect"></select>
      </div>

      <div class="block">
        <label>Policy Host</label>
        <input id="policyHostInput" value="127.0.0.1" />
      </div>

      <div class="block">
        <label>Policy Port</label>
        <input id="policyPortInput" type="number" value="8001" />
      </div>

      <div class="block">
        <label>Device</label>
        <input id="policyDeviceInput" value="cuda" />
      </div>

      <div class="block">
        <label>Rollout Instruction</label>
        <textarea id="rolloutInstructionInput" rows="4"></textarea>
      </div>

      <div class="block">
        <label>Rollout Step Size</label>
        <input id="rolloutStepsInput" type="number" min="1" max="100" value="1" />
      </div>

      <div class="block">
        <label>Policy Server</label>
        <div id="policyServerStatus" class="meta">Not started.</div>
      </div>

      <div class="block">
        <div id="policyButtonRow"></div>
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
const policyBackendSelect = root.querySelector<HTMLSelectElement>("#policyBackendSelect");
const policyModelSelect = root.querySelector<HTMLSelectElement>("#policyModelSelect");
const policyHostInput = root.querySelector<HTMLInputElement>("#policyHostInput");
const policyPortInput = root.querySelector<HTMLInputElement>("#policyPortInput");
const policyDeviceInput = root.querySelector<HTMLInputElement>("#policyDeviceInput");
const rolloutInstructionInput = root.querySelector<HTMLTextAreaElement>("#rolloutInstructionInput");
const rolloutStepsInput = root.querySelector<HTMLInputElement>("#rolloutStepsInput");
const policyServerStatus = root.querySelector<HTMLDivElement>("#policyServerStatus");
const policyButtonRow = root.querySelector<HTMLDivElement>("#policyButtonRow");

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
  !buttonRow ||
  !policyBackendSelect ||
  !policyModelSelect ||
  !policyHostInput ||
  !policyPortInput ||
  !policyDeviceInput ||
  !rolloutInstructionInput ||
  !rolloutStepsInput ||
  !policyServerStatus ||
  !policyButtonRow
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
const policyBackendSelectEl = policyBackendSelect;
const policyModelSelectEl = policyModelSelect;
const policyHostInputEl = policyHostInput;
const policyPortInputEl = policyPortInput;
const policyDeviceInputEl = policyDeviceInput;
const rolloutInstructionInputEl = rolloutInstructionInput;
const rolloutStepsInputEl = rolloutStepsInput;
const policyServerStatusEl = policyServerStatus;
const policyButtonRowEl = policyButtonRow;

manifestInputEl.value = getManifestUrlFromLocation();

const loadBtn = makeButton("Load env");
const refreshBtn = makeButton("Refresh exports");
const resetBtn = makeButton("Reset");
const inspectModeBtn = makeButton("Inspect");
const setupModeBtn = makeButton("Setup");
const startPolicyServerBtn = makeButton("Start XVLA");
const stopPolicyServerBtn = makeButton("Stop XVLA");
const attachPolicyBtn = makeButton("Attach Policy");
const rolloutStepBtn = makeButton("Policy Step");
const rolloutRunBtn = makeButton("Run Rollout");
const rolloutStopBtn = makeButton("Stop Rollout");

buttonRowEl.append(loadBtn, refreshBtn, resetBtn);
modeRowEl.append(inspectModeBtn, setupModeBtn);
policyButtonRowEl.append(
  startPolicyServerBtn,
  stopPolicyServerBtn,
  attachPolicyBtn,
  rolloutStepBtn,
  rolloutRunBtn,
  rolloutStopBtn,
);

const viewer = new MujocoDebugViewer(viewerElEl);

let loaded: LoadedBundle | null = null;
let liveSession: LiveSession | null = null;
let raf = 0;
let exportEntries: ExportIndexEntry[] = [];
let currentMode: "inspect" | "setup" = "inspect";
let selectedMutableObject = "";
let selectedJointName = "";
let objectStateCache: ObjectStateResponse | null = null;
const pressedKeys = new Set<string>();
let setupTickHandle = 0;
let setupMutationInFlight = false;
let policyCatalog: Record<string, PolicyBackendEntry> = {};
let policyServerPollHandle = 0;
let rolloutRunning = false;
let rolloutRequestInFlight = false;
let rolloutStreamAbort: AbortController | null = null;

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
    ? `live: connected\nmutable objects: ${liveSession.scene.mutable_objects.length}\npolicy: ${liveSession.policy?.backend ?? "none"}`
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

function selectedPolicyBackend(): string {
  return policyBackendSelectEl.value || "xvla";
}

function selectedPolicyModelId(): string | null {
  return policyModelSelectEl.value || null;
}

function selectedRolloutInstruction(): string {
  const value = rolloutInstructionInputEl.value.trim();
  return value || liveSession?.instruction || loaded?.manifest.instruction || "";
}

function selectedRolloutStepCount(): number {
  const parsed = Number.parseInt(rolloutStepsInputEl.value, 10);
  return Number.isFinite(parsed) ? Math.min(Math.max(parsed, 1), 100) : 1;
}

function updatePolicyServerUi(status: PolicyServerStatus | null) {
  if (!status || !status.running) {
    policyServerStatusEl.textContent = "Not started.";
    return;
  }

  const health = status.healthy ? "healthy" : "starting";
  policyServerStatusEl.textContent =
    `backend: ${status.backend ?? "-"}\n` +
    `model: ${status.model_id ?? "-"}\n` +
    `address: ${status.host ?? "-"}:${status.port ?? "-"}\n` +
    `pid: ${status.pid ?? "-"}\n` +
    `status: ${health}`;
}

function fillPolicyBackendOptions() {
  policyBackendSelectEl.innerHTML = "";
  for (const [backend, entry] of Object.entries(policyCatalog)) {
    const opt = document.createElement("option");
    opt.value = backend;
    opt.textContent = entry.label || backend;
    policyBackendSelectEl.appendChild(opt);
  }
  if (!policyBackendSelectEl.value && policyBackendSelectEl.options.length > 0) {
    policyBackendSelectEl.value = policyBackendSelectEl.options[0].value;
  }
  fillPolicyModelOptions();
}

function fillPolicyModelOptions() {
  const backend = selectedPolicyBackend();
  const models = policyCatalog[backend]?.models ?? [];
  const previous = policyModelSelectEl.value;
  policyModelSelectEl.innerHTML = "";
  for (const model of models) {
    const opt = document.createElement("option");
    opt.value = model.id;
    opt.textContent = model.label || model.id;
    policyModelSelectEl.appendChild(opt);
  }
  if (models.some((model) => model.id === previous)) {
    policyModelSelectEl.value = previous;
  } else if (models.length > 0) {
    policyModelSelectEl.value = models[0].id;
  }
}

function applySelectedCamera() {
  if (!loaded) return;
  const name = cameraSelectEl.value;
  if (!name) return;
  viewer.setActiveCameraByName(name);
}

function stopLoop() {
  if (raf) {
    cancelAnimationFrame(raf);
    raf = 0;
  }
}

function frame() {
  if (!loaded) {
    return;
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
      instruction: live.instruction,
      policy: live.policy ?? null,
    };
    await viewer.setLoadedScene(loaded);
    applySimState(live.state);
    rolloutInstructionInputEl.value = live.instruction;
    fillManifestInfo();
    await loadSelectedObjectState();
    applySelectedCamera();
    setStatus("Env loaded and synced with live backend", "ok");
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

async function refreshPolicyCatalog() {
  const response = await fetch("/api/policies/catalog");
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to fetch policy catalog: ${response.status} ${message}`);
  }
  const payload = (await response.json()) as PolicyCatalogResponse;
  policyCatalog = payload.policies;
  fillPolicyBackendOptions();
}

async function fetchPolicyServerStatus(): Promise<PolicyServerStatus> {
  const response = await fetch("/api/policy/server");
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to fetch policy server status: ${response.status} ${message}`);
  }
  return (await response.json()) as PolicyServerStatus;
}

async function refreshPolicyServerStatus() {
  try {
    const status = await fetchPolicyServerStatus();
    updatePolicyServerUi(status);
  } catch {
    updatePolicyServerUi(null);
  }
}

async function startPolicyServer() {
  const response = await fetch("/api/policy/server/start", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      backend: selectedPolicyBackend(),
      model_id: selectedPolicyModelId(),
      host: policyHostInputEl.value.trim() || "127.0.0.1",
      port: Number.parseInt(policyPortInputEl.value, 10) || 8001,
      device: policyDeviceInputEl.value.trim() || "cuda",
    }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to start policy server: ${response.status} ${message}`);
  }
  await refreshPolicyServerStatus();
}

async function stopPolicyServer() {
  const response = await fetch("/api/policy/server/stop", {
    method: "POST",
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to stop policy server: ${response.status} ${message}`);
  }
  await refreshPolicyServerStatus();
}

function shouldRefreshSelectedObjectState(): boolean {
  return currentMode === "setup" && !!selectedMutableObject;
}

async function attachPolicy() {
  if (!loaded || !liveSession) {
    throw new Error("Load an env before attaching a policy.");
  }
  const response = await fetch("/api/policy/configure", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      backend: selectedPolicyBackend(),
      model_id: selectedPolicyModelId(),
      host: policyHostInputEl.value.trim() || "127.0.0.1",
      port: Number.parseInt(policyPortInputEl.value, 10) || 8001,
      timeout_s: 30,
      restart_if_needed: true,
    }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to attach policy: ${response.status} ${message}`);
  }
  const payload = (await response.json()) as {
    policy: PolicyConfigResponse | null;
    state: BackendState;
  };
  liveSession = {
    ...liveSession,
    instruction: selectedRolloutInstruction() || liveSession.instruction,
    policy: payload.policy,
  };
  applySimState(payload.state);
  fillManifestInfo();
  if (shouldRefreshSelectedObjectState()) {
    await loadSelectedObjectState();
  }
}

async function rolloutStep(): Promise<RolloutStepResponse> {
  const response = await fetch("/api/rollout/step", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      instruction: selectedRolloutInstruction(),
      num_steps: selectedRolloutStepCount(),
    }),
  });
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to run rollout step: ${response.status} ${message}`);
  }
  return (await response.json()) as RolloutStepResponse;
}

async function fetchObjectState(objectName: string): Promise<ObjectStateResponse> {
  const response = await fetch(`/api/mutable/object/${encodeURIComponent(objectName)}`);
  if (!response.ok) {
    const message = await response.text();
    throw new Error(`Failed to fetch object state: ${response.status} ${message}`);
  }
  return (await response.json()) as ObjectStateResponse;
}

async function mutateJoint(
  objectName: string,
  jointName: string,
  value: number,
): Promise<{ state: BackendState; objectState: ObjectStateResponse }> {
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
  const payload = (await response.json()) as {
    state: BackendState;
    object_state: ObjectStateResponse;
  };
  return {
    state: payload.state,
    objectState: payload.object_state,
  };
}

async function mutatePose(
  objectName: string,
  pos: number[],
  quat: number[],
): Promise<{ state: BackendState; objectState: ObjectStateResponse }> {
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
  const payload = (await response.json()) as {
    state: BackendState;
    object_state: ObjectStateResponse;
  };
  return {
    state: payload.state,
    objectState: payload.object_state,
  };
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

function applyQposState(state: QposState) {
  if (!loaded) {
    return;
  }

  if (Array.isArray(state.qpos)) {
    const n = Math.min(state.qpos.length, loaded.data.qpos.length);
    for (let i = 0; i < n; i++) {
      loaded.data.qpos[i] = state.qpos[i];
    }
  }

  loaded.mujoco.mj_forward(loaded.model, loaded.data);
  viewer.syncFromPhysics();
}

async function runPolicyStep() {
  if (!loaded || !liveSession) {
    throw new Error("Load an env before running rollout.");
  }
  if (rolloutRequestInFlight) {
    return;
  }
  rolloutRequestInFlight = true;
  try {
    const payload = await rolloutStep();
    applySimState(payload.state);
    if (shouldRefreshSelectedObjectState()) {
      await loadSelectedObjectState();
    }
    const rewards = payload.steps.map((step) => step.reward.toFixed(3)).join(", ");
    setStatus(payload.done ? `Rollout finished (${rewards})` : `Rollout stepped (${rewards})`, payload.done ? "warn" : "ok");
  } finally {
    rolloutRequestInFlight = false;
  }
}

async function runRolloutLoop() {
  if (rolloutRunning) {
    return;
  }
  rolloutRunning = true;
  rolloutStreamAbort = new AbortController();
  setStatus("Policy rollout running", "ok");

  try {
    const response = await fetch("/api/rollout/stream", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        instruction: selectedRolloutInstruction(),
        num_steps: selectedRolloutStepCount(),
      }),
      signal: rolloutStreamAbort.signal,
    });

    if (!response.ok) {
      const message = await response.text();
      throw new Error(`Failed to stream rollout: ${response.status} ${message}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Rollout stream body is unavailable");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    while (rolloutRunning) {
      const chunk = await reader.read();
      if (chunk.done) {
        break;
      }

      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;

        const event = JSON.parse(trimmed) as RolloutStreamEvent;
        if (event.type === "error") {
          throw new Error(event.error);
        }

        if (event.type === "step") {
          applyQposState(event.state);
          setStatus(
            event.done
              ? `Policy rollout finished (${event.step.reward.toFixed(3)})`
              : `Policy rollout step ${event.index + 1} (${event.step.reward.toFixed(3)})`,
            event.done ? "warn" : "ok"
          );
          if (event.done) {
            rolloutRunning = false;
            break;
          }
        }

        if (event.type === "done") {
          applyQposState(event.state);
          rolloutRunning = false;
          setStatus(
            event.done
              ? `Policy rollout finished (${event.steps_sent} steps)`
              : `Policy rollout stopped (${event.steps_sent} steps)`,
            "warn"
          );
          break;
        }
      }
    }
  } catch (err) {
    const aborted = err instanceof DOMException && err.name === "AbortError";
    if (!aborted) {
      rolloutRunning = false;
      throw err;
    }
  } finally {
    rolloutRunning = false;
    rolloutStreamAbort = null;
    if (shouldRefreshSelectedObjectState()) {
      await loadSelectedObjectState();
    }
  }
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
      const payload = await mutateJoint(sceneObject.name, selectedJointName, nextValue);
      objectStateCache = payload.objectState;
      applySimState(payload.state);
      renderSetupControls();
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
      const payload = await mutatePose(sceneObject.name, pos, cached.quat);
      objectStateCache = payload.objectState;
      applySimState(payload.state);
      renderSetupControls();
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

resetBtn.onclick = () => {
  if (!liveSession) {
    return;
  }
  void (async () => {
    try {
      const state = await resetLiveSession();
      applySimState(state);
      await loadSelectedObjectState();
      setStatus("Reset to LIBERO initial state", "ok");
      applySelectedCamera();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
    }
  })();
};

startPolicyServerBtn.onclick = () => {
  void (async () => {
    try {
      await startPolicyServer();
      setStatus("Started XVLA server", "ok");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
      await refreshPolicyServerStatus();
    }
  })();
};

stopPolicyServerBtn.onclick = () => {
  void (async () => {
    try {
      await stopPolicyServer();
      setStatus("Stopped XVLA server", "warn");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
    }
  })();
};

attachPolicyBtn.onclick = () => {
  void (async () => {
    try {
      await attachPolicy();
      setStatus("Attached policy to live session", "ok");
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
    }
  })();
};

rolloutStepBtn.onclick = () => {
  void (async () => {
    try {
      await runPolicyStep();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
    }
  })();
};

rolloutRunBtn.onclick = () => {
  void (async () => {
    try {
      await runRolloutLoop();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setStatus(msg, "err");
      console.error(err);
    }
  })();
};

rolloutStopBtn.onclick = () => {
  rolloutRunning = false;
  rolloutStreamAbort?.abort();
  rolloutStreamAbort = null;
  setStatus("Policy rollout stopped", "warn");
};

cameraSelectEl.onchange = () => {
  applySelectedCamera();
};

exportSelectEl.onchange = () => {
  manifestInputEl.value = exportSelectEl.value;
};

policyBackendSelectEl.onchange = () => {
  fillPolicyModelOptions();
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
    await refreshPolicyCatalog();
    await refreshPolicyServerStatus();
    setMode("inspect");
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    setStatus(msg, "err");
    console.error(err);
  }
})();

policyServerPollHandle = window.setInterval(() => {
  void refreshPolicyServerStatus();
}, 2000);
