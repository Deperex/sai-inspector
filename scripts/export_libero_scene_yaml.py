#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from sai_inspector.envs.libero_env import LiberoExportEnv
from sai_inspector.envs.scene_metadata import build_scene_metadata
from export_libero_env import build_export_env


def export_scene_yaml(
    *,
    env_name: str,
    task_id: int,
    cfg: dict[str, Any],
    env: LiberoExportEnv,
    out_root: str,
) -> Path:
    scene_info = build_scene_metadata(
        env_name=env_name,
        benchmark_name=cfg["benchmark_name"],
        task_id=task_id,
        instruction=env.instruction,
        env=env,
        mutable_objects=cfg.get("mutable_objects", []),
    )

    out_dir = Path(out_root) / env_name / f"task_{task_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene.yaml"
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(scene_info, f, sort_keys=False)

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--registry", default="src/sai_inspector/registry/libero.yaml")
    parser.add_argument("--out", default="exports")
    args = parser.parse_args()

    cfg, env = build_export_env(args.env, args.task_id, args.registry)
    out_path = export_scene_yaml(
        env_name=args.env,
        task_id=args.task_id,
        cfg=cfg,
        env=env,
        out_root=args.out,
    )
    print(f"Wrote scene metadata to: {out_path}")


if __name__ == "__main__":
    main()
