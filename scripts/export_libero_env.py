#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import yaml

from sai_inspector.envs.libero_env import LiberoExportEnv
from sai_inspector.envs.export_assets import copy_and_rewrite_assets


def load_env_config(env_name: str, registry_path: str) -> dict:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    envs = data.get("envs", {})
    if env_name not in envs:
        raise KeyError(f"Env '{env_name}' not found in {registry_path}")

    return envs[env_name]


def build_export_env(env_name: str, task_id: int, registry_path: str) -> tuple[dict[str, Any], LiberoExportEnv]:
    cfg = load_env_config(env_name, registry_path)
    env = LiberoExportEnv(
        benchmark_name=cfg["benchmark_name"],
        task_id=task_id,
        camera_names=cfg.get("camera_names", ["agentview"]),
        camera_heights=cfg.get("render", {}).get("height", 256),
        camera_widths=cfg.get("render", {}).get("width", 256),
    )
    return cfg, env


def list_camera_names(model: mujoco.MjModel) -> list[str]:
    names = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        names.append(name if name else f"camera_{i}")
    return names


def export_env_artifacts(
    *,
    env_name: str,
    task_id: int,
    cfg: dict[str, Any],
    env: LiberoExportEnv,
    out_root: str,
) -> Path:
    out_dir = Path(out_root) / env_name / f"task_{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    xml_path = out_dir / "scene.xml"
    mujoco.mj_saveLastXML(str(xml_path), env.model)

    state = {
        "qpos": [float(x) for x in env.data.qpos],
        "qvel": [float(x) for x in env.data.qvel],
    }
    (out_dir / "state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")

    manifest = {
        "env_name": env_name,
        "benchmark_name": cfg["benchmark_name"],
        "task_id": task_id,
        "instruction": env.instruction,
        "xml": "scene.xml",
        "state": "state.json",
        "scene": "scene.yaml",
        "cameras": list_camera_names(env.model),
        "assets": copy_and_rewrite_assets(xml_path, out_dir),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Exported to: {out_dir}")
    print(f"Instruction: {env.instruction}")
    print(f"Cameras: {manifest['cameras']}")
    return out_dir


def export_task(env_name: str, task_id: int, registry_path: str, out_root: str) -> None:
    cfg, env = build_export_env(env_name, task_id, registry_path)
    export_env_artifacts(
        env_name=env_name,
        task_id=task_id,
        cfg=cfg,
        env=env,
        out_root=out_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--registry", default="src/sai_inspector/registry/libero.yaml")
    parser.add_argument("--out", default="exports")
    args = parser.parse_args()

    export_task(
        env_name=args.env,
        task_id=args.task_id,
        registry_path=args.registry,
        out_root=args.out,
    )


if __name__ == "__main__":
    main()
