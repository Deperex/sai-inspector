function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

type MujocoModule = {
  FS: any;
  MjModel: {
    from_xml_path(path: string): any;
  };
  MjData: new (model: any) => any;
  mj_step(model: any, data: any): void;
  mj_resetData(model: any, data: any): void;
  mj_forward(model: any, data: any): void;
};

async function loadMujocoModule(): Promise<MujocoModule> {
  console.log("MuJoCo init: import start");

  const mod = await import("@mujoco/mujoco/mt");
  console.log("MuJoCo init: import done");

  const factory = (mod as any).default ?? mod;
  console.log("MuJoCo init: factory ready");

  const instance = await factory({
    locateFile(path: string) {
      console.log("MuJoCo init: locateFile", path);
      if (path.endsWith(".wasm")) return "/mujoco_mt.wasm";
      return path;
    },
  });

  console.log("MuJoCo init: factory resolved");
  return instance;
}

async function fetchJson<T>(url: string): Promise<T> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  return await r.json();
}

async function fetchText(url: string): Promise<string> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  return await r.text();
}

async function fetchBytes(url: string): Promise<Uint8Array> {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}

const MAX_RUNTIME_TEXTURE_DIMENSION = 512;

function isImageAsset(path: string): boolean {
  return /\.(png|jpg|jpeg|webp)$/i.test(path);
}

async function fetchDownscaledImageBytes(url: string, maxDimension: number): Promise<Uint8Array> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Failed to fetch ${url}: ${response.status}`);

  const blob = await response.blob();
  const bitmap = await createImageBitmap(blob);

  if (Math.max(bitmap.width, bitmap.height) <= maxDimension) {
    bitmap.close();
    return new Uint8Array(await blob.arrayBuffer());
  }

  const scale = maxDimension / Math.max(bitmap.width, bitmap.height);
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) {
    bitmap.close();
    return new Uint8Array(await blob.arrayBuffer());
  }

  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  const mimeType = blob.type === "image/jpeg" ? "image/jpeg" : "image/png";
  const resizedBlob = await new Promise<Blob | null>((resolve) => {
    canvas.toBlob((value) => resolve(value), mimeType, 0.9);
  });

  if (!resizedBlob) {
    return new Uint8Array(await blob.arrayBuffer());
  }
  return new Uint8Array(await resizedBlob.arrayBuffer());
}

function collectXmlFileRefs(xmlText: string): string[] {
  const refs = new Set<string>();
  const re = /file="([^"]+)"/g;
  for (let m = re.exec(xmlText); m; m = re.exec(xmlText)) {
    const p = m[1]?.trim();
    if (!p) continue;
    if (p.startsWith("http://") || p.startsWith("https://")) continue;
    refs.add(p.replace(/^\/+/, ""));
  }
  return [...refs];
}

function manifestBase(manifestUrl: string): string {
  return manifestUrl.replace(/\/[^/]*$/, "");
}

function ensureDir(FS: any, dir: string) {
  const parts = dir.split("/").filter(Boolean);
  let cur = "";
  for (const p of parts) {
    cur += `/${p}`;
    try {
      FS.mkdir(cur);
    } catch {}
  }
}

function writeBinaryFile(FS: any, path: string, data: Uint8Array) {
  const dir = path.slice(0, path.lastIndexOf("/"));
  if (dir) ensureDir(FS, dir);
  FS.writeFile(path, data);
}

function writeTextFile(FS: any, path: string, text: string) {
  const dir = path.slice(0, path.lastIndexOf("/"));
  if (dir) ensureDir(FS, dir);
  FS.writeFile(path, text);
}

type BundleManifest = {
  env_name: string;
  benchmark_name: string;
  task_id: number;
  instruction: string;
  cameras: string[];
  xml: string;
  state: string;
  assets: string[];
};

export type LoadedBundle = {
  mujoco: MujocoModule;
  manifest: BundleManifest;
  model: any;
  data: any;
  state: any;
  xmlText: string;
  baseUrl: string;
};

export async function loadBundle(manifestUrl: string): Promise<LoadedBundle> {
  const mujoco = await withTimeout(loadMujocoModule(), 15000, "MuJoCo init");
  const baseUrl = manifestBase(manifestUrl);
  const manifest = await fetchJson<BundleManifest>(manifestUrl);
  const [state, xmlText] = await Promise.all([
    fetchJson<any>(`${baseUrl}/${manifest.state}`),
    fetchText(`${baseUrl}/${manifest.xml}`),
  ]);
  const runtimeXmlText = xmlText;

  const fsRoot = "/bundle";
  const xmlFsPath = `${fsRoot}/${manifest.xml.replace(/^\/+/, "")}`;
  const xmlRefs = collectXmlFileRefs(runtimeXmlText);

  for (const assetPath of xmlRefs) {
    const assetUrl = `${baseUrl}/${assetPath}`;
    const data = isImageAsset(assetPath)
      ? await fetchDownscaledImageBytes(assetUrl, MAX_RUNTIME_TEXTURE_DIMENSION)
      : await fetchBytes(assetUrl);
    writeBinaryFile(mujoco.FS, `${fsRoot}/${assetPath.replace(/^\/+/, "")}`, data);
  }

  writeTextFile(mujoco.FS, xmlFsPath, runtimeXmlText);
  const model = mujoco.MjModel.from_xml_path(xmlFsPath);

  const data = new mujoco.MjData(model);
  return {
    mujoco,
    manifest,
    model,
    data,
    state,
    xmlText,
    baseUrl,
  };
}
