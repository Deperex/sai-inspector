from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sai_inspector.envs.libero_env import LiberoExportEnv
from sai_inspector.envs.scene_metadata import build_scene_metadata


def load_env_config(env_name: str, registry_path: str) -> dict[str, Any]:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    envs = data.get("envs", {})
    if env_name not in envs:
        raise KeyError(f"Env '{env_name}' not found in {registry_path}")

    return dict(envs[env_name])


def snapshot_state(env: LiberoExportEnv) -> dict[str, Any]:
    return {
        "qpos": [float(x) for x in env.data.qpos],
        "qvel": [float(x) for x in env.data.qvel],
    }


@dataclass
class LiveSession:
    env_name: str
    task_id: int
    registry_path: str
    cfg: dict[str, Any]
    env: LiberoExportEnv
    scene: dict[str, Any]


class SessionManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._session: LiveSession | None = None

    def start(self, env_name: str, task_id: int, registry_path: str) -> LiveSession:
        cfg = load_env_config(env_name, registry_path)
        env = LiberoExportEnv(
            benchmark_name=cfg["benchmark_name"],
            task_id=task_id,
            camera_names=cfg.get("camera_names", ["agentview"]),
            camera_heights=cfg.get("render", {}).get("height", 256),
            camera_widths=cfg.get("render", {}).get("width", 256),
        )
        scene = build_scene_metadata(
            env_name=env_name,
            benchmark_name=cfg["benchmark_name"],
            task_id=task_id,
            instruction=env.instruction,
            env=env,
            mutable_objects=cfg.get("mutable_objects", []),
        )
        session = LiveSession(
            env_name=env_name,
            task_id=task_id,
            registry_path=registry_path,
            cfg=cfg,
            env=env,
            scene=scene,
        )
        with self._lock:
            old_session = self._session
            self._session = session
        if old_session is not None:
            old_session.env._control.close()
        return session

    def get(self) -> LiveSession:
        with self._lock:
            session = self._session
        if session is None:
            raise HTTPException(status_code=404, detail="No active LIBERO session")
        return session

    def reset(self) -> dict[str, Any]:
        session = self.get()
        session.env.reset()
        return snapshot_state(session.env)


class StartSessionRequest(BaseModel):
    env: str
    task_id: int
    registry: str = "src/sai_inspector/registry/libero.yaml"


class JointMutationRequest(BaseModel):
    object_name: str = Field(alias="object")
    joint_name: str = Field(alias="joint")
    value: float
    clamp: bool = True

    model_config = {"populate_by_name": True}


class PoseMutationRequest(BaseModel):
    object_name: str = Field(alias="object")
    pos: list[float]
    quat: list[float]

    model_config = {"populate_by_name": True}


app = FastAPI(title="SAI Inspector Backend")
sessions = SessionManager()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session/start")
def start_session(req: StartSessionRequest) -> dict[str, Any]:
    registry_path = str(Path(req.registry))
    try:
        session = sessions.start(req.env, req.task_id, registry_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "env_name": session.env_name,
        "task_id": session.task_id,
        "instruction": session.env.instruction,
        "scene": session.scene,
        "state": snapshot_state(session.env),
    }


@app.get("/api/session")
def get_session() -> dict[str, Any]:
    session = sessions.get()
    return {
        "env_name": session.env_name,
        "task_id": session.task_id,
        "instruction": session.env.instruction,
    }


@app.get("/api/session/scene")
def get_scene() -> dict[str, Any]:
    return sessions.get().scene


@app.get("/api/session/state")
def get_state() -> dict[str, Any]:
    return snapshot_state(sessions.get().env)


@app.post("/api/session/reset")
def reset_session() -> dict[str, Any]:
    return {"state": sessions.reset()}


@app.get("/api/mutable/object/{object_name}")
def get_object_state(object_name: str) -> dict[str, Any]:
    session = sessions.get()
    try:
        return session.env.get_object_state(object_name)
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/mutable/joint")
def set_joint(req: JointMutationRequest) -> dict[str, Any]:
    session = sessions.get()
    try:
        value = session.env.set_object_joint(
            req.object_name,
            req.joint_name,
            req.value,
            clamp=req.clamp,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "object": req.object_name,
        "joint": req.joint_name,
        "value": value,
        "state": snapshot_state(session.env),
    }


@app.post("/api/mutable/pose")
def set_pose(req: PoseMutationRequest) -> dict[str, Any]:
    session = sessions.get()
    try:
        pose = session.env.set_object_pose(
            req.object_name,
            pos=req.pos,
            quat=req.quat,
        )
    except (KeyError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "object": req.object_name,
        "pose": pose,
        "state": snapshot_state(session.env),
    }
