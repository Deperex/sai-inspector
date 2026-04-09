from __future__ import annotations

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path


def _safe_relative_asset_path(src: Path) -> Path:
    """
    Convert an absolute source asset path into a clean relative export path.
    """

    src_str = str(src)

    # LIBERO assets
    if "/third_party/envs/libero/libero/libero/assets/" in src_str:
        rel = src_str.split("/third_party/envs/libero/libero/libero/assets/", 1)[1]
        return Path("assets") / rel

    # robosuite assets from site-packages
    if "/site-packages/robosuite/models/assets/" in src_str:
        rel = src_str.split("/site-packages/robosuite/models/assets/", 1)[1]
        return Path("assets") / "robosuite" / rel

    # fallback
    return Path("assets") / src.name


def copy_and_rewrite_assets(xml_path: str | Path, export_root: str | Path) -> list[str]:
    xml_path = Path(xml_path)
    export_root = Path(export_root)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.attrib.pop("meshdir", None)
        compiler.attrib.pop("texturedir", None)

    copied_files: list[str] = []
    seen: set[tuple[str, str]] = set()

    for elem in root.iter():
        file_attr = elem.attrib.get("file")
        if not file_attr:
            continue

        src = Path(file_attr)

        # Resolve relative references against XML location
        if not src.is_absolute():
            src = (xml_path.parent / src).resolve()

        if not src.exists():
            print(f"[warning] missing asset: {src}")
            continue

        rel_export_path = _safe_relative_asset_path(src)
        dst = export_root / rel_export_path

        key = (str(src), str(dst))
        if key not in seen:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
            copied_files.append(str(rel_export_path))
            seen.add(key)

        # Rewrite XML file attribute to browser-friendly relative path
        elem.set("file", str(rel_export_path).replace("\\", "/"))

    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    return copied_files
