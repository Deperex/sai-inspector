import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { OBJLoader } from "three/examples/jsm/loaders/OBJLoader.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";
import { parseSceneXml, type ParsedSceneXml } from "./parseSceneXml";

type LoadedScene = {
  mujoco: any;
  model: any;
  data: any;
  xmlText: string;
  baseUrl: string;
};

const MJ_GEOM_PLANE = 0;
const MJ_GEOM_HFIELD = 1;
const MJ_GEOM_SPHERE = 2;
const MJ_GEOM_CAPSULE = 3;
const MJ_GEOM_ELLIPSOID = 4;
const MJ_GEOM_CYLINDER = 5;
const MJ_GEOM_BOX = 6;
const MJ_GEOM_MESH = 7;
const MJ_GEOM_SDF = 8;

type RenderGeom = {
  object: THREE.Object3D;
  geomId: number;
  isPlane?: boolean;
};

type ResolvedVisual = {
  color: THREE.Color;
  opacity: number;
  textureFile?: string;
  textureRepeat?: [number, number];
  modelTexId?: number;
};

const USE_TEXTURE_MAPS = true;

function quatFromMat3Array(m: ArrayLike<number>): THREE.Quaternion {
  const mat4 = new THREE.Matrix4();
  mat4.set(
    m[0], m[1], m[2], 0,
    m[3], m[4], m[5], 0,
    m[6], m[7], m[8], 0,
    0, 0, 0, 1
  );
  const q = new THREE.Quaternion();
  q.setFromRotationMatrix(mat4);
  return q;
}

function makeMaterialFromColor(color: THREE.Color, opacity = 1.0, texture: THREE.Texture | null = null) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness: 0.8,
    metalness: 0.05,
    transparent: opacity < 0.999,
    opacity,
    map: texture,
  });
}

function colorFromModel(model: any, geomId: number): THREE.Color {
  const base = geomId * 4;
  let r = Number(model.geom_rgba[base + 0] ?? 0.7);
  let g = Number(model.geom_rgba[base + 1] ?? 0.7);
  let b = Number(model.geom_rgba[base + 2] ?? 0.7);

  const matId = Number(model.geom_matid?.[geomId] ?? -1);
  if (matId >= 0 && model.mat_rgba) {
    const mb = matId * 4;
    const mr = Number(model.mat_rgba[mb + 0]);
    const mg = Number(model.mat_rgba[mb + 1]);
    const mbb = Number(model.mat_rgba[mb + 2]);
    if (
      Number.isFinite(mr) &&
      Number.isFinite(mg) &&
      Number.isFinite(mbb)
    ) {
      r = mr;
      g = mg;
      b = mbb;
    }
  }

  return new THREE.Color(r, g, b);
}

function cloneTextureForVisual(
  texture: THREE.Texture | null,
  repeat?: [number, number]
): THREE.Texture | null {
  if (!texture) return null;
  const tex = texture.clone();
  tex.colorSpace = texture.colorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(repeat?.[0] ?? 1, repeat?.[1] ?? 1);
  tex.needsUpdate = true;
  return tex;
}

function opacityFromModel(model: any, geomId: number): number {
  const ga = Number(model.geom_rgba[geomId * 4 + 3] ?? 1.0);
  const matId = Number(model.geom_matid?.[geomId] ?? -1);
  if (matId >= 0 && model.mat_rgba && Math.abs(ga - 1.0) < 1e-6) {
    const ma = Number(model.mat_rgba[matId * 4 + 3]);
    if (Number.isFinite(ma)) return ma;
  }
  return ga;
}

function geomName(model: any, mujoco: any, geomId: number): string | null {
  try {
    const byApi = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geomId) ?? null;
    if (byApi) return byApi;
  } catch {
    // Fall through to raw-name-table lookup below.
  }
  try {
    const adr = Number(model.name_geomadr?.[geomId] ?? -1);
    if (adr >= 0 && model.names) {
      const byTable = readCString(model.names, adr);
      if (byTable) return byTable;
    }
  } catch {
    // Ignore and return null below.
  }
  return null;
}

function primitiveFallback(model: any, geomId: number, mat: THREE.Material): THREE.Object3D {
  const geomType = model.geom_type[geomId];
  const s0 = model.geom_size[geomId * 3 + 0] ?? 0.05;
  const s1 = model.geom_size[geomId * 3 + 1] ?? s0;
  const s2 = model.geom_size[geomId * 3 + 2] ?? s0;

  if (geomType === MJ_GEOM_PLANE) {
    return new THREE.Mesh(
      new THREE.PlaneGeometry(8, 8),
      mat
    );
  }

  if (geomType === MJ_GEOM_SPHERE) {
    return new THREE.Mesh(new THREE.SphereGeometry(Math.max(s0, 1e-4), 24, 16), mat);
  }

  if (geomType === MJ_GEOM_BOX) {
    return new THREE.Mesh(
      new THREE.BoxGeometry(
        Math.max(2 * s0, 1e-4),
        Math.max(2 * s1, 1e-4),
        Math.max(2 * s2, 1e-4)
      ),
      mat
    );
  }

  if (geomType === MJ_GEOM_CYLINDER) {
    const mesh = new THREE.Mesh(
      new THREE.CylinderGeometry(
        Math.max(s0, 1e-4),
        Math.max(s0, 1e-4),
        Math.max(2 * s1, 1e-4),
        24
      ),
      mat
    );
    // MuJoCo primitives are z-axis aligned; Three cylinders are y-axis aligned.
    mesh.rotateX(Math.PI / 2);
    const root = new THREE.Group();
    root.add(mesh);
    return root;
  }

  if (geomType === MJ_GEOM_CAPSULE) {
    const root = new THREE.Group();
    const group = new THREE.Group();

    const cyl = new THREE.Mesh(
      new THREE.CylinderGeometry(
        Math.max(s0, 1e-4),
        Math.max(s0, 1e-4),
        Math.max(2 * s1, 1e-4),
        24
      ),
      mat
    );
    cyl.rotateX(Math.PI / 2);
    group.add(cyl);

    const capA = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(s0, 1e-4), 24, 16),
      mat
    );
    capA.position.z = s1;
    group.add(capA);

    const capB = new THREE.Mesh(
      new THREE.SphereGeometry(Math.max(s0, 1e-4), 24, 16),
      mat
    );
    capB.position.z = -s1;
    group.add(capB);

    root.add(group);
    return root;
  }

  if (geomType === MJ_GEOM_ELLIPSOID) {
    const mesh = new THREE.Mesh(new THREE.SphereGeometry(1, 24, 16), mat);
    mesh.scale.set(Math.max(s0, 1e-4), Math.max(s1, 1e-4), Math.max(s2, 1e-4));
    return mesh;
  }

  const mesh = new THREE.Mesh(new THREE.SphereGeometry(Math.max(s0, s1, s2, 0.03), 16, 12), mat);
  return mesh;
}

function readCString(buffer: ArrayLike<number>, adr: number): string {
  if (!Number.isFinite(adr) || adr < 0) return "";
  let out = "";
  for (let i = adr; i < adr + 4096; i++) {
    const c = Number(buffer[i] ?? 0);
    if (!Number.isFinite(c) || c === 0) break;
    out += String.fromCharCode(c);
  }
  return out;
}

function readTableVec3(table: any, row: number): [number, number, number] | null {
  const entry = table?.[row];
  if (entry && typeof entry === "object" && typeof entry.length === "number" && entry.length >= 3) {
    const x = Number(entry[0]);
    const y = Number(entry[1]);
    const z = Number(entry[2]);
    if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) return [x, y, z];
    return null;
  }

  const x = Number(table?.[row * 3 + 0]);
  const y = Number(table?.[row * 3 + 1]);
  const z = Number(table?.[row * 3 + 2]);
  if (Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z)) return [x, y, z];
  return null;
}

function readTableVec2(table: any, row: number): [number, number] | null {
  const entry = table?.[row];
  if (entry && typeof entry === "object" && typeof entry.length === "number" && entry.length >= 2) {
    const x = Number(entry[0]);
    const y = Number(entry[1]);
    if (Number.isFinite(x) && Number.isFinite(y)) return [x, y];
    return null;
  }

  const x = Number(table?.[row * 2 + 0]);
  const y = Number(table?.[row * 2 + 1]);
  if (Number.isFinite(x) && Number.isFinite(y)) return [x, y];
  return null;
}

function meshFromModelData(model: any, geomId: number): THREE.Object3D | null {
  const meshId = Number(model.geom_dataid?.[geomId] ?? -1);
  if (!Number.isFinite(meshId) || meshId < 0) return null;

  const vertAdr = Number(model.mesh_vertadr?.[meshId] ?? -1);
  const vertNum = Number(model.mesh_vertnum?.[meshId] ?? 0);
  const faceAdr = Number(model.mesh_faceadr?.[meshId] ?? -1);
  const faceNum = Number(model.mesh_facenum?.[meshId] ?? 0);
  if (vertAdr < 0 || faceAdr < 0 || vertNum <= 0 || faceNum <= 0) return null;

  const texcoordAdr = Number(model.mesh_texcoordadr?.[meshId] ?? -1);
  const texcoordNum = Number(model.mesh_texcoordnum?.[meshId] ?? 0);
  const hasTexcoords =
    texcoordAdr >= 0 &&
    texcoordNum > 0 &&
    model.mesh_texcoord &&
    model.mesh_facetexcoord;

  const positions = new Float32Array(vertNum * 3);
  for (let i = 0; i < vertNum; i++) {
    const v = readTableVec3(model.mesh_vert, vertAdr + i);
    if (!v) return null;
    positions[i * 3 + 0] = v[0];
    positions[i * 3 + 1] = v[1];
    positions[i * 3 + 2] = v[2];
  }

  if (hasTexcoords) {
    const expandedPositions: number[] = [];
    const expandedUvs: number[] = [];

    for (let i = 0; i < faceNum; i++) {
      const f = readTableVec3(model.mesh_face, faceAdr + i);
      const ft = readTableVec3(model.mesh_facetexcoord, faceAdr + i);
      if (!f || !ft) continue;

      const verts = [Math.trunc(f[0]), Math.trunc(f[1]), Math.trunc(f[2])];
      const texcoords = [Math.trunc(ft[0]), Math.trunc(ft[1]), Math.trunc(ft[2])];

      let valid = true;
      for (let j = 0; j < 3; j++) {
        if (verts[j] < 0 || verts[j] >= vertNum) valid = false;
        if (texcoords[j] < 0 || texcoords[j] >= texcoordNum) valid = false;
      }
      if (!valid) continue;
      if (verts[0] === verts[1] || verts[1] === verts[2] || verts[0] === verts[2]) continue;

      for (let j = 0; j < 3; j++) {
        const v = readTableVec3(model.mesh_vert, vertAdr + verts[j]);
        const uv = readTableVec2(model.mesh_texcoord, texcoordAdr + texcoords[j]);
        if (!v || !uv) {
          valid = false;
          break;
        }
        expandedPositions.push(v[0], v[1], v[2]);
        expandedUvs.push(uv[0], 1 - uv[1]);
      }
      if (!valid) {
        expandedPositions.splice(-9);
        expandedUvs.splice(-6);
      }
    }

    if (expandedPositions.length >= 9 && expandedUvs.length >= 6) {
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(expandedPositions, 3));
      geom.setAttribute("uv", new THREE.Float32BufferAttribute(expandedUvs, 2));
      geom.computeVertexNormals();
      return new THREE.Mesh(
        geom,
        makeMaterialFromColor(colorFromModel(model, geomId), opacityFromModel(model, geomId))
      );
    }
  }

  const valid: number[] = [];
  for (let i = 0; i < faceNum; i++) {
    const f = readTableVec3(model.mesh_face, faceAdr + i);
    if (!f) continue;
    const a = Math.trunc(f[0]);
    const b = Math.trunc(f[1]);
    const c = Math.trunc(f[2]);
    if (
      a < 0 || a >= vertNum ||
      b < 0 || b >= vertNum ||
      c < 0 || c >= vertNum
    ) {
      continue;
    }
    if (a === b || b === c || a === c) continue;
    valid.push(a, b, c);
  }
  if (valid.length < 3) return null;

  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geom.setIndex(new THREE.BufferAttribute(new Uint32Array(valid), 1));
  geom.computeVertexNormals();
  return new THREE.Mesh(geom, makeMaterialFromColor(colorFromModel(model, geomId), opacityFromModel(model, geomId)));
}

function applyMaterialRecursive(
  obj: THREE.Object3D,
  color: THREE.Color,
  opacity = 1,
  texture: THREE.Texture | null = null,
  textureRepeat?: [number, number]
) {
  obj.traverse((child) => {
    const mesh = child as THREE.Mesh;
    if (mesh.isMesh) {
      const textureForMesh = cloneTextureForVisual(texture, textureRepeat);
      mesh.material = new THREE.MeshStandardMaterial({
        color,
        roughness: 0.8,
        metalness: 0.05,
        transparent: opacity < 0.999,
        opacity,
        map: textureForMesh,
      });
      mesh.castShadow = false;
      mesh.receiveShadow = true;
    }
  });
}

function objectHasAnyUv(obj: THREE.Object3D): boolean {
  let hasUv = false;
  obj.traverse((child) => {
    if (hasUv) return;
    const mesh = child as THREE.Mesh;
    if (!mesh.isMesh) return;
    const geometry = mesh.geometry as THREE.BufferGeometry | undefined;
    const uvAttr = geometry?.getAttribute?.("uv");
    if (uvAttr && uvAttr.count > 0) {
      hasUv = true;
    }
  });
  return hasUv;
}

async function loadObj(url: string): Promise<THREE.Object3D> {
  const loader = new OBJLoader();
  return await loader.loadAsync(url);
}

async function loadStl(
  url: string,
  color: THREE.Color,
  opacity = 1,
  texture: THREE.Texture | null = null,
  textureRepeat?: [number, number]
): Promise<THREE.Object3D> {
  const loader = new STLLoader();
  const geometry = await loader.loadAsync(url);
  geometry.computeVertexNormals();
  return new THREE.Mesh(geometry, makeMaterialFromColor(color, opacity, cloneTextureForVisual(texture, textureRepeat)));
}

function joinUrl(baseUrl: string, relPath: string): string {
  return `${baseUrl}/${relPath.replace(/^\/+/, "")}`;
}

function applyMeshAssetTransform(obj: THREE.Object3D, model: any, meshId: number) {
  if (!Number.isFinite(meshId) || meshId < 0) return;
  const px = Number(model.mesh_pos?.[meshId * 3 + 0] ?? 0);
  const py = Number(model.mesh_pos?.[meshId * 3 + 1] ?? 0);
  const pz = Number(model.mesh_pos?.[meshId * 3 + 2] ?? 0);
  const qw = Number(model.mesh_quat?.[meshId * 4 + 0] ?? 1);
  const qx = Number(model.mesh_quat?.[meshId * 4 + 1] ?? 0);
  const qy = Number(model.mesh_quat?.[meshId * 4 + 2] ?? 0);
  const qz = Number(model.mesh_quat?.[meshId * 4 + 3] ?? 0);
  if (Number.isFinite(px) && Number.isFinite(py) && Number.isFinite(pz)) {
    obj.position.set(px, py, pz);
  }
  if (Number.isFinite(qw) && Number.isFinite(qx) && Number.isFinite(qy) && Number.isFinite(qz)) {
    obj.quaternion.set(qx, qy, qz, qw);
  }
}

function shouldRenderGeom(parsed: ParsedSceneXml, model: any, mujoco: any, geomId: number): boolean {
  const name = geomName(model, mujoco, geomId) ?? "";
  const info = parsed.geomVisualsByName[name];

  if (name.includes("collision")) return false;
  if (info?.group === 1) return true;
  if (name.includes("visual") || name.endsWith("_vis")) return true;

  const ct = Number(model.geom_contype?.[geomId] ?? 0);
  const ca = Number(model.geom_conaffinity?.[geomId] ?? 0);
  if (ct === 0 && ca === 0) return true;

  return false;
}

const textureCache = new Map<string, Promise<THREE.Texture | null>>();
const averageColorCache = new Map<string, Promise<THREE.Color | null>>();
const modelTextureCache = new Map<string, THREE.Texture | null>();
const modelTextureAverageColorCache = new Map<string, THREE.Color | null>();
const MAX_TEXTURE_DIMENSION = 1024;
const textureLoader = new THREE.TextureLoader();

function makeConfiguredTexture(image: TexImageSource): THREE.Texture {
  const tex = new THREE.Texture(image);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.needsUpdate = true;
  return tex;
}

async function loadDownscaledTexture(url: string): Promise<THREE.Texture | null> {
  try {
    const response = await fetch(url);
    if (!response.ok) return null;

    const blob = await response.blob();
    const bitmap = await createImageBitmap(blob);

    if (
      bitmap.width <= MAX_TEXTURE_DIMENSION &&
      bitmap.height <= MAX_TEXTURE_DIMENSION
    ) {
      return makeConfiguredTexture(bitmap);
    }

    const scale = Math.min(
      1,
      MAX_TEXTURE_DIMENSION / Math.max(bitmap.width, bitmap.height)
    );
    const width = Math.max(1, Math.round(bitmap.width * scale));
    const height = Math.max(1, Math.round(bitmap.height * scale));

    const canvas = document.createElement("canvas");
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      bitmap.close();
      return null;
    }

    ctx.drawImage(bitmap, 0, 0, width, height);
    bitmap.close();
    return makeConfiguredTexture(canvas);
  } catch {
    return null;
  }
}

async function loadTextureCached(baseUrl: string, relPath: string): Promise<THREE.Texture | null> {
  const url = joinUrl(baseUrl, relPath);
  const cached = textureCache.get(url);
  if (cached) return cached;
  const p = textureLoader.loadAsync(url)
    .then((tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
      tex.needsUpdate = true;
      return tex;
    })
    .catch(async () => {
      return await loadDownscaledTexture(url);
    });
  textureCache.set(url, p);
  return await p;
}

async function loadAverageTextureColor(baseUrl: string, relPath: string): Promise<THREE.Color | null> {
  const url = joinUrl(baseUrl, relPath);
  const cached = averageColorCache.get(url);
  if (cached) return cached;

  const p = (async () => {
    try {
      const response = await fetch(url);
      if (!response.ok) return null;

      const blob = await response.blob();
      const bitmap = await createImageBitmap(blob);

      const sampleSize = 16;
      const canvas = document.createElement("canvas");
      canvas.width = sampleSize;
      canvas.height = sampleSize;

      const ctx = canvas.getContext("2d", { willReadFrequently: true });
      if (!ctx) {
        bitmap.close();
        return null;
      }

      ctx.drawImage(bitmap, 0, 0, sampleSize, sampleSize);
      bitmap.close();

      const { data } = ctx.getImageData(0, 0, sampleSize, sampleSize);
      let r = 0;
      let g = 0;
      let b = 0;
      let count = 0;

      for (let i = 0; i < data.length; i += 4) {
        const alpha = data[i + 3] / 255;
        if (alpha <= 0) continue;
        r += data[i + 0] * alpha;
        g += data[i + 1] * alpha;
        b += data[i + 2] * alpha;
        count += alpha;
      }

      if (count <= 0) return null;
      return new THREE.Color(r / (255 * count), g / (255 * count), b / (255 * count));
    } catch {
      return null;
    }
  })();

  averageColorCache.set(url, p);
  return await p;
}

function loadTextureFromModel(model: any, texId: number): THREE.Texture | null {
  if (!Number.isFinite(texId) || texId < 0) return null;
  const key = `${Number(model.ntex ?? 0)}:${texId}`;
  if (modelTextureCache.has(key)) return modelTextureCache.get(key) ?? null;

  const width = Number(model.tex_width?.[texId] ?? 0);
  const height = Number(model.tex_height?.[texId] ?? 0);
  const channels = Number(model.tex_nchannel?.[texId] ?? 0);
  const adr = Number(model.tex_adr?.[texId] ?? -1);
  if (width <= 0 || height <= 0 || channels <= 0 || adr < 0 || !model.tex_data) {
    modelTextureCache.set(key, null);
    return null;
  }

  const pixelCount = width * height;
  const length = pixelCount * channels;
  const raw = new Uint8Array(length);
  for (let i = 0; i < length; i++) {
    raw[i] = Number(model.tex_data[adr + i] ?? 0);
  }

  const rgba = new Uint8Array(pixelCount * 4);
  for (let i = 0; i < pixelCount; i++) {
    const src = i * channels;
    const dst = i * 4;
    if (channels >= 3) {
      rgba[dst + 0] = raw[src + 0];
      rgba[dst + 1] = raw[src + 1];
      rgba[dst + 2] = raw[src + 2];
      rgba[dst + 3] = channels >= 4 ? raw[src + 3] : 255;
    } else if (channels === 2) {
      rgba[dst + 0] = raw[src + 0];
      rgba[dst + 1] = raw[src + 0];
      rgba[dst + 2] = raw[src + 0];
      rgba[dst + 3] = raw[src + 1];
    } else if (channels === 1) {
      rgba[dst + 0] = raw[src + 0];
      rgba[dst + 1] = raw[src + 0];
      rgba[dst + 2] = raw[src + 0];
      rgba[dst + 3] = 255;
    } else {
      rgba[dst + 0] = 0;
      rgba[dst + 1] = 0;
      rgba[dst + 2] = 0;
      rgba[dst + 3] = 255;
    }
  }

  let texture: THREE.Texture | null = new THREE.DataTexture(
    rgba,
    width,
    height,
    THREE.RGBAFormat,
    THREE.UnsignedByteType
  );

  if (!texture) {
    modelTextureCache.set(key, null);
    return null;
  }

  texture.colorSpace = THREE.SRGBColorSpace;
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.flipY = true;
  texture.needsUpdate = true;
  modelTextureCache.set(key, texture);
  return texture;
}

function averageColorFromModelTexture(model: any, texId: number): THREE.Color | null {
  if (!Number.isFinite(texId) || texId < 0) return null;
  const key = `${Number(model.ntex ?? 0)}:${texId}`;
  if (modelTextureAverageColorCache.has(key)) {
    return modelTextureAverageColorCache.get(key) ?? null;
  }

  const width = Number(model.tex_width?.[texId] ?? 0);
  const height = Number(model.tex_height?.[texId] ?? 0);
  const channels = Number(model.tex_nchannel?.[texId] ?? 0);
  const adr = Number(model.tex_adr?.[texId] ?? -1);
  if (width <= 0 || height <= 0 || channels <= 0 || adr < 0 || !model.tex_data) {
    modelTextureAverageColorCache.set(key, null);
    return null;
  }

  const length = width * height * channels;
  let r = 0;
  let g = 0;
  let b = 0;
  let count = 0;
  for (let i = 0; i < length; i += channels) {
    const rr = Number(model.tex_data[adr + i + 0] ?? 0);
    const gg = Number(model.tex_data[adr + i + Math.min(1, channels - 1)] ?? rr);
    const bb = Number(model.tex_data[adr + i + Math.min(2, channels - 1)] ?? rr);
    const aa = channels >= 4 ? Number(model.tex_data[adr + i + 3] ?? 255) / 255 : 1;
    if (aa <= 0) continue;
    r += rr * aa;
    g += gg * aa;
    b += bb * aa;
    count += aa;
  }

  const color = count > 0
    ? new THREE.Color(r / (255 * count), g / (255 * count), b / (255 * count))
    : null;
  modelTextureAverageColorCache.set(key, color);
  return color;
}

function resolveVisual(
  parsed: ParsedSceneXml,
  geomInfo: ParsedSceneXml["geomVisualsByName"][string] | undefined,
  model: any,
  geomId: number
): ResolvedVisual {
  let color = colorFromModel(model, geomId);
  let opacity = opacityFromModel(model, geomId);
  let textureFile: string | undefined;
  let textureRepeat: [number, number] | undefined;
  let modelTexId: number | undefined;

  const matName = geomInfo?.materialName;
  let materialHasExplicitRgba = false;
  const matId = Number(model.geom_matid?.[geomId] ?? -1);
  if (matId >= 0) {
    const texId = Number(model.mat_texid?.[matId] ?? -1);
    if (texId >= 0) {
      modelTexId = texId;
    }
    const tx = Number(model.mat_texrepeat?.[matId * 2 + 0] ?? NaN);
    const ty = Number(model.mat_texrepeat?.[matId * 2 + 1] ?? NaN);
    if (Number.isFinite(tx) && Number.isFinite(ty)) {
      textureRepeat = [tx, ty];
    }
  }
  if (matName) {
    const mat = parsed.materialAssetsByName[matName];
    if (mat?.rgba) {
      color = new THREE.Color(mat.rgba[0], mat.rgba[1], mat.rgba[2]);
      opacity = mat.rgba[3];
      materialHasExplicitRgba = true;
    }
    if (mat?.textureName) {
      textureFile = parsed.textureAssetsByName[mat.textureName]?.file;
    }
    if (mat?.texrepeat) {
      textureRepeat = mat.texrepeat;
    }
  }

  let geomHasExplicitRgba = false;
  if (geomInfo?.rgba) {
    color = new THREE.Color(geomInfo.rgba[0], geomInfo.rgba[1], geomInfo.rgba[2]);
    opacity = geomInfo.rgba[3];
    geomHasExplicitRgba = true;
  }

  if (USE_TEXTURE_MAPS && textureFile && !materialHasExplicitRgba && !geomHasExplicitRgba) {
    // Textured MuJoCo materials are usually intended to display the image as-is.
    color = new THREE.Color(1, 1, 1);
  }

  return { color, opacity, textureFile, textureRepeat, modelTexId };
}

async function createMeshObject(
  parsed: ParsedSceneXml,
  baseUrl: string,
  model: any,
  mujoco: any,
  geomId: number
): Promise<THREE.Object3D | null> {
  const name = geomName(model, mujoco, geomId);
  const geomInfo = name ? parsed.geomVisualsByName[name] : undefined;
  const meshName = geomInfo?.meshName;
  const meshInfo = meshName ? parsed.meshAssetsByName[meshName] : undefined;

  const meshId = Number(model.geom_dataid?.[geomId] ?? -1);
  const meshPathAdr = Number(model.mesh_pathadr?.[meshId] ?? -1);
  const meshPath = readCString(model.paths ?? [], meshPathAdr);

  const meshFile = meshInfo?.file ?? meshPath;
  if (!meshFile) return null;

  const fileLower = meshFile.toLowerCase();
  const url = joinUrl(baseUrl, meshFile);
  const visual = resolveVisual(parsed, geomInfo, model, geomId);
  let texture: THREE.Texture | null = visual.modelTexId != null ? loadTextureFromModel(model, visual.modelTexId) : null;
  let color = visual.color;
  let fallbackAverageColor: THREE.Color | null =
    visual.modelTexId != null ? averageColorFromModelTexture(model, visual.modelTexId) : null;
  if (!texture && visual.textureFile) {
    if (USE_TEXTURE_MAPS) {
      texture = await loadTextureCached(baseUrl, visual.textureFile);
      if (!texture) {
        const averageColor = await loadAverageTextureColor(baseUrl, visual.textureFile);
        if (averageColor) {
          color = averageColor;
          fallbackAverageColor = averageColor;
        }
        console.warn("Texture map load failed for mesh geom", {
          geomName: name,
          geomId,
          textureFile: visual.textureFile,
        });
      }
    } else {
      const averageColor = await loadAverageTextureColor(baseUrl, visual.textureFile);
      if (averageColor) color = averageColor;
      fallbackAverageColor = averageColor;
    }
  }

  // Prefer MuJoCo-compiled mesh data for geometric fidelity with current sim state.
  // File-based OBJ/STL fallback is only used when mesh buffers are unavailable.
  const mjMeshObj = meshFromModelData(model, geomId);
  if (mjMeshObj) {
    const hasUv = objectHasAnyUv(mjMeshObj);
    if (texture && !hasUv) {
      if (!fallbackAverageColor && visual.textureFile) {
        fallbackAverageColor = await loadAverageTextureColor(baseUrl, visual.textureFile);
      }
      if (fallbackAverageColor) {
        color = fallbackAverageColor;
        texture = null;
      }
    }
    applyMaterialRecursive(mjMeshObj, color, visual.opacity, texture, visual.textureRepeat);
    return mjMeshObj;
  }

  let obj: THREE.Object3D | null = null;

  if (fileLower.endsWith(".obj")) {
    obj = await loadObj(url);
    const hasUv = objectHasAnyUv(obj);
    if (texture && !hasUv) {
      if (!fallbackAverageColor && visual.textureFile) {
        fallbackAverageColor = await loadAverageTextureColor(baseUrl, visual.textureFile);
      }
      if (fallbackAverageColor) {
        color = fallbackAverageColor;
        texture = null;
      }
    }
    applyMaterialRecursive(obj, color, visual.opacity, texture, visual.textureRepeat);
    if (meshId >= 0) applyMeshAssetTransform(obj, model, meshId);
  } else if (fileLower.endsWith(".stl")) {
    obj = await loadStl(url, color, visual.opacity, texture, visual.textureRepeat);
    if (meshId >= 0) applyMeshAssetTransform(obj, model, meshId);
  } else if (fileLower.endsWith(".msh")) {
    obj = meshFromModelData(model, geomId);
    if (obj) {
      const hasUv = objectHasAnyUv(obj);
      if (texture && !hasUv) {
        if (!fallbackAverageColor && visual.textureFile) {
          fallbackAverageColor = await loadAverageTextureColor(baseUrl, visual.textureFile);
        }
        if (fallbackAverageColor) {
          color = fallbackAverageColor;
          texture = null;
        }
      }
      applyMaterialRecursive(obj, color, visual.opacity, texture, visual.textureRepeat);
    }
  } else {
    return null;
  }
  if (!obj) return null;

  if (meshInfo?.scale) {
    obj.scale.multiply(new THREE.Vector3(meshInfo.scale[0], meshInfo.scale[1], meshInfo.scale[2]));
  } else if (meshId >= 0) {
    const sx = Number(model.mesh_scale?.[meshId * 3 + 0] ?? 1);
    const sy = Number(model.mesh_scale?.[meshId * 3 + 1] ?? 1);
    const sz = Number(model.mesh_scale?.[meshId * 3 + 2] ?? 1);
    if (Number.isFinite(sx) && Number.isFinite(sy) && Number.isFinite(sz)) {
      obj.scale.multiply(new THREE.Vector3(sx, sy, sz));
    }
  }
  return obj;
}

async function createPrimitiveObject(
  parsed: ParsedSceneXml,
  baseUrl: string,
  model: any,
  mujoco: any,
  geomId: number
): Promise<THREE.Object3D> {
  const name = geomName(model, mujoco, geomId);
  const geomInfo = name ? parsed.geomVisualsByName[name] : undefined;
  const visual = resolveVisual(parsed, geomInfo, model, geomId);
  let texture: THREE.Texture | null = visual.modelTexId != null ? loadTextureFromModel(model, visual.modelTexId) : null;
  let color = visual.color;
  let fallbackAverageColor: THREE.Color | null =
    visual.modelTexId != null ? averageColorFromModelTexture(model, visual.modelTexId) : null;
  if (!texture && visual.textureFile) {
    if (USE_TEXTURE_MAPS) {
      texture = await loadTextureCached(baseUrl, visual.textureFile);
      if (!texture) {
        const averageColor = await loadAverageTextureColor(baseUrl, visual.textureFile);
        if (averageColor) {
          color = averageColor;
          fallbackAverageColor = averageColor;
        }
        console.warn("Texture map load failed for primitive geom", {
          geomName: name,
          geomId,
          textureFile: visual.textureFile,
        });
      }
    } else {
      const averageColor = await loadAverageTextureColor(baseUrl, visual.textureFile);
      if (averageColor) color = averageColor;
      fallbackAverageColor = averageColor;
    }
  }
  if (!texture && fallbackAverageColor) {
    color = fallbackAverageColor;
  }
  const mat = makeMaterialFromColor(
    color,
    visual.opacity,
    cloneTextureForVisual(texture, visual.textureRepeat)
  );
  return primitiveFallback(model, geomId, mat);
}

export class MujocoDebugViewer {
  private container: HTMLElement;
  private renderer: THREE.WebGLRenderer;
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private controls: OrbitControls;
  private loaded: LoadedScene | null = null;
  private renderGeoms: RenderGeom[] = [];
  private overlay: HTMLDivElement;

  constructor(container: HTMLElement) {
    this.container = container;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x101318);
    this.scene.up.set(0, 0, 1);

    this.camera = new THREE.PerspectiveCamera(
      50,
      Math.max(container.clientWidth, 1) / Math.max(container.clientHeight, 1),
      0.01,
      100
    );
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(2.0, 1.8, 2.2);

    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 0.8, 0);
    this.controls.update();

    const amb = new THREE.AmbientLight(0xffffff, 0.8);
    this.scene.add(amb);

    const dir = new THREE.DirectionalLight(0xffffff, 1.0);
    dir.position.set(3, 4, 2);
    this.scene.add(dir);

    const grid = new THREE.GridHelper(8, 40, 0x444444, 0x222222);
    grid.rotateX(Math.PI / 2);
    this.scene.add(grid);

    const axes = new THREE.AxesHelper(0.5);
    this.scene.add(axes);

    this.overlay = document.createElement("div");
    this.overlay.className = "overlay";
    this.overlay.textContent = "No scene loaded";
    container.appendChild(this.overlay);

    window.addEventListener("resize", this.onResize);
  }

  dispose() {
    window.removeEventListener("resize", this.onResize);
    this.controls.dispose();
    this.renderer.dispose();
  }

  private onResize = () => {
    const w = Math.max(this.container.clientWidth, 1);
    const h = Math.max(this.container.clientHeight, 1);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  };

  async setLoadedScene(loaded: LoadedScene) {
    this.clearScene();
    this.loaded = loaded;

    const { model, mujoco, xmlText, baseUrl } = loaded;
    const parsed = parseSceneXml(xmlText);

    for (let i = 0; i < model.ngeom; i++) {
      if (!shouldRenderGeom(parsed, model, mujoco, i)) continue;

      let obj: THREE.Object3D | null = null;

      if (model.geom_type[i] === MJ_GEOM_MESH) {
        obj = await createMeshObject(parsed, baseUrl, model, mujoco, i);
        if (!obj) {
          obj = meshFromModelData(model, i);
        }
      }

      if (!obj) {
        obj = await createPrimitiveObject(parsed, baseUrl, model, mujoco, i);
      }

      this.scene.add(obj);
      this.renderGeoms.push({
        object: obj,
        geomId: i,
        isPlane: model.geom_type[i] === MJ_GEOM_PLANE,
      });
    }

    this.syncFromPhysics();
    this.fitCameraToLoadedScene();
    this.render();
  }

  syncFromPhysics() {
    if (!this.loaded) return;

    const { model, data } = this.loaded;

    for (const entry of this.renderGeoms) {
      const geomId = entry.geomId;
      const obj = entry.object;

      const posBase = geomId * 3;
      obj.position.set(
        data.geom_xpos[posBase + 0] ?? 0,
        data.geom_xpos[posBase + 1] ?? 0,
        data.geom_xpos[posBase + 2] ?? 0
      );

      const matBase = geomId * 9;
      const q = quatFromMat3Array([
        data.geom_xmat[matBase + 0], data.geom_xmat[matBase + 1], data.geom_xmat[matBase + 2],
        data.geom_xmat[matBase + 3], data.geom_xmat[matBase + 4], data.geom_xmat[matBase + 5],
        data.geom_xmat[matBase + 6], data.geom_xmat[matBase + 7], data.geom_xmat[matBase + 8],
      ]);

      obj.quaternion.copy(q);

    }

    this.overlay.textContent =
      `Bodies: ${model.nbody}\n` +
      `Geoms: ${model.ngeom}\n` +
      `Joints: ${model.njnt}\n` +
      `Actuators: ${model.nu}`;
  }

  render() {
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  setActiveCameraByName(name: string): boolean {
    if (!this.loaded) return false;
    const { model, data, mujoco } = this.loaded;
    let camId = -1;
    try {
      camId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name);
    } catch {
      camId = -1;
    }
    if (!Number.isFinite(camId) || camId < 0) return false;

    const p = camId * 3;
    const m = camId * 9;
    const pos = new THREE.Vector3(
      Number(data.cam_xpos[p + 0] ?? 0),
      Number(data.cam_xpos[p + 1] ?? 0),
      Number(data.cam_xpos[p + 2] ?? 0)
    );
    const q = quatFromMat3Array([
      Number(data.cam_xmat[m + 0] ?? 1), Number(data.cam_xmat[m + 1] ?? 0), Number(data.cam_xmat[m + 2] ?? 0),
      Number(data.cam_xmat[m + 3] ?? 0), Number(data.cam_xmat[m + 4] ?? 1), Number(data.cam_xmat[m + 5] ?? 0),
      Number(data.cam_xmat[m + 6] ?? 0), Number(data.cam_xmat[m + 7] ?? 0), Number(data.cam_xmat[m + 8] ?? 1),
    ]);

    this.camera.position.copy(pos);
    this.camera.quaternion.copy(q);
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(q);
    this.controls.target.copy(pos.clone().add(dir.multiplyScalar(0.5)));
    this.controls.update();
    this.render();
    return true;
  }

  private clearScene() {
    for (const entry of this.renderGeoms) {
      this.scene.remove(entry.object);
    }
    this.renderGeoms = [];
  }

  private fitCameraToLoadedScene() {
    if (this.renderGeoms.length === 0) return;

    const box = new THREE.Box3();
    for (const entry of this.renderGeoms) {
      box.expandByObject(entry.object);
    }

    if (!Number.isFinite(box.min.x) || !Number.isFinite(box.max.x)) return;

    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const radius = Math.max(size.x, size.y, size.z, 0.5);
    const dist = radius * 1.8;

    this.controls.target.copy(center);
    this.camera.position.set(center.x + dist, center.y + dist * 0.6, center.z + dist);
    this.camera.near = Math.max(0.001, dist / 500);
    this.camera.far = Math.max(100, dist * 20);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }
}
