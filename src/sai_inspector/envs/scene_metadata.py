from __future__ import annotations

from typing import Any

import mujoco

from sai_inspector.envs.libero_env import LiberoExportEnv


def joint_type_name(joint_type: int) -> str:
    return {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }.get(int(joint_type), f"unknown_{joint_type}")


def infer_object_mobility(joints: list[dict[str, Any]]) -> str:
    if any(joint["type"] == "free" for joint in joints):
        return "free"
    if any(joint["type"] in {"slide", "hinge", "ball"} for joint in joints):
        return "articulated"
    return "static"


def joint_names_for_object(model: mujoco.MjModel, object_name: str) -> list[str]:
    prefix = f"{object_name}_"
    names: list[str] = []
    for joint_id in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if joint_name and joint_name.startswith(prefix):
            names.append(joint_name)
    return names


def build_joint_catalog(
    model: mujoco.MjModel,
    object_name: str,
    configured_mutable: bool,
) -> dict[str, Any]:
    joints: list[dict[str, Any]] = []

    for joint_name in joint_names_for_object(model, object_name):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        joint_type = joint_type_name(int(model.jnt_type[joint_id]))
        joint_info: dict[str, Any] = {
            "name": joint_name,
            "type": joint_type,
            "qpos_addr": int(model.jnt_qposadr[joint_id]),
            "dof_addr": int(model.jnt_dofadr[joint_id]),
        }

        if joint_type in {"slide", "hinge"}:
            joint_info["axis"] = [float(v) for v in model.jnt_axis[joint_id]]
            if bool(model.jnt_limited[joint_id]):
                joint_info["range"] = [float(v) for v in model.jnt_range[joint_id]]

        joints.append(joint_info)

    return {
        "name": object_name,
        "mobility": infer_object_mobility(joints),
        "mutable_candidate": bool(joints),
        "configured_mutable": configured_mutable,
        "joints": joints,
    }


def build_scene_metadata(
    *,
    env_name: str,
    benchmark_name: str,
    task_id: int,
    instruction: str,
    env: LiberoExportEnv,
    mutable_objects: list[str] | set[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    mutable_set = set(mutable_objects or [])

    object_entries: list[dict[str, Any]] = []
    for kind, mapping in (
        ("object", env.env.objects_dict),
        ("fixture", env.env.fixtures_dict),
    ):
        for object_name in sorted(mapping.keys()):
            entry = build_joint_catalog(
                env.model,
                object_name=object_name,
                configured_mutable=object_name in mutable_set,
            )
            entry["kind"] = kind
            object_entries.append(entry)

    known_names = {entry["name"] for entry in object_entries}
    unknown_mutable_objects = sorted(name for name in mutable_set if name not in known_names)

    joints_by_name: dict[str, dict[str, Any]] = {}
    for entry in object_entries:
        for joint in entry["joints"]:
            joints_by_name[joint["name"]] = {
                "object_name": entry["name"],
                "kind": entry["kind"],
                "mobility": entry["mobility"],
                "configured_mutable": entry["configured_mutable"],
                **joint,
            }

    return {
        "env_name": env_name,
        "benchmark_name": benchmark_name,
        "task_id": int(task_id),
        "instruction": instruction,
        "mutable_objects": sorted(mutable_set),
        "unknown_mutable_objects": unknown_mutable_objects,
        "objects": object_entries,
        "joints_by_name": joints_by_name,
    }
