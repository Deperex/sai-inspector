export type MeshAssetInfo = {
  name: string;
  file: string;
  scale?: [number, number, number];
};

export type GeomVisualInfo = {
  type: string;
  meshName?: string;
  materialName?: string;
  rgba?: [number, number, number, number];
  group?: number;
  contype?: number;
  conaffinity?: number;
};

export type TextureAssetInfo = {
  name: string;
  file: string;
};

export type MaterialAssetInfo = {
  name: string;
  textureName?: string;
  rgba?: [number, number, number, number];
  texrepeat?: [number, number];
  texuniform?: boolean;
};

export type ParsedSceneXml = {
  meshAssetsByName: Record<string, MeshAssetInfo>;
  textureAssetsByName: Record<string, TextureAssetInfo>;
  materialAssetsByName: Record<string, MaterialAssetInfo>;
  geomVisualsByName: Record<string, GeomVisualInfo>;
};

function parseVec(input: string | null, n: number): number[] | undefined {
  if (!input) return undefined;
  const vals = input
    .trim()
    .split(/\s+/)
    .map((x) => Number(x))
    .filter((x) => Number.isFinite(x));

  if (vals.length < n) return undefined;
  return vals.slice(0, n);
}

export function parseSceneXml(xmlText: string): ParsedSceneXml {
  const doc = new DOMParser().parseFromString(xmlText, "application/xml");

  const meshAssetsByName: Record<string, MeshAssetInfo> = {};
  const textureAssetsByName: Record<string, TextureAssetInfo> = {};
  const materialAssetsByName: Record<string, MaterialAssetInfo> = {};
  const geomVisualsByName: Record<string, GeomVisualInfo> = {};

  const meshElems = Array.from(doc.querySelectorAll("asset mesh"));
  for (const elem of meshElems) {
    const name = elem.getAttribute("name");
    const file = elem.getAttribute("file");
    if (!name || !file) continue;

    const scaleRaw = parseVec(elem.getAttribute("scale"), 3);
    meshAssetsByName[name] = {
      name,
      file,
      scale: scaleRaw ? [scaleRaw[0], scaleRaw[1], scaleRaw[2]] : undefined,
    };
  }

  const textureElems = Array.from(doc.querySelectorAll("asset texture"));
  for (const elem of textureElems) {
    const name = elem.getAttribute("name");
    const file = elem.getAttribute("file");
    if (!name || !file) continue;
    textureAssetsByName[name] = { name, file };
  }

  const materialElems = Array.from(doc.querySelectorAll("asset material"));
  for (const elem of materialElems) {
    const name = elem.getAttribute("name");
    if (!name) continue;
    const rgbaRaw = parseVec(elem.getAttribute("rgba"), 4);
    const texrepeatRaw = parseVec(elem.getAttribute("texrepeat"), 2);
    materialAssetsByName[name] = {
      name,
      textureName: elem.getAttribute("texture") ?? undefined,
      rgba: rgbaRaw ? [rgbaRaw[0], rgbaRaw[1], rgbaRaw[2], rgbaRaw[3]] : undefined,
      texrepeat: texrepeatRaw ? [texrepeatRaw[0], texrepeatRaw[1]] : undefined,
      texuniform: elem.getAttribute("texuniform") === "true",
    };
  }

  const geomElems = Array.from(doc.querySelectorAll("geom"));
  for (const elem of geomElems) {
    const name = elem.getAttribute("name");
    if (!name) continue;

    const rgbaRaw = parseVec(elem.getAttribute("rgba"), 4);

    geomVisualsByName[name] = {
      type: elem.getAttribute("type") ?? "unknown",
      meshName: elem.getAttribute("mesh") ?? undefined,
      materialName: elem.getAttribute("material") ?? undefined,
      rgba: rgbaRaw ? [rgbaRaw[0], rgbaRaw[1], rgbaRaw[2], rgbaRaw[3]] : undefined,
      group: elem.getAttribute("group") ? Number(elem.getAttribute("group")) : undefined,
      contype: elem.getAttribute("contype") ? Number(elem.getAttribute("contype")) : undefined,
      conaffinity: elem.getAttribute("conaffinity")
        ? Number(elem.getAttribute("conaffinity"))
        : undefined,
    };
  }

  return {
    meshAssetsByName,
    textureAssetsByName,
    materialAssetsByName,
    geomVisualsByName,
  };
}
