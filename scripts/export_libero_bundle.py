#!/usr/bin/env python3
from __future__ import annotations

import argparse

from export_libero_env import build_export_env, export_env_artifacts
from export_libero_scene_yaml import export_scene_yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--registry", default="src/sai_inspector/registry/libero.yaml")
    parser.add_argument("--out", default="exports")
    args = parser.parse_args()

    cfg, env = build_export_env(args.env, args.task_id, args.registry)
    export_env_artifacts(
        env_name=args.env,
        task_id=args.task_id,
        cfg=cfg,
        env=env,
        out_root=args.out,
    )
    scene_path = export_scene_yaml(
        env_name=args.env,
        task_id=args.task_id,
        cfg=cfg,
        env=env,
        out_root=args.out,
    )
    print(f"Wrote scene metadata to: {scene_path}")


if __name__ == "__main__":
    main()
