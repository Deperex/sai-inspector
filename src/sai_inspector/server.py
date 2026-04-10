from __future__ import annotations

import logging
import os
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from sai_inspector.envs.libero_env import LiberoExportEnv
from sai_inspector.envs.scene_metadata import build_scene_metadata
from sai_inspector.policies.xvla import XVLAClient, XVLAPolicyAdapter

logger = logging.getLogger("sai_inspector.server")


def load_env_config(env_name: str, registry_path: str) -> dict[str, Any]:
    with open(registry_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    envs = data.get("envs", {})
    if env_name not in envs:
        raise KeyError(f"Env '{env_name}' not found in {registry_path}")

    return dict(envs[env_name])


def load_policy_catalog(config_path: str = "src/sai_inspector/policies/config.yaml") -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return dict(data.get("policies", {}))


def resolve_policy_model(backend: str, model_id: str | None, config_path: str = "src/sai_inspector/policies/config.yaml") -> dict[str, Any]:
    catalog = load_policy_catalog(config_path)
    backend_entry = catalog.get(backend)
    if backend_entry is None:
        raise HTTPException(status_code=404, detail=f"Policy backend '{backend}' not found in {config_path}")

    models = backend_entry.get("models", [])
    if not models:
        raise HTTPException(status_code=404, detail=f"Policy backend '{backend}' has no configured models")

    if model_id is None:
        return dict(models[0])

    for model in models:
        if model.get("id") == model_id:
            return dict(model)

    raise HTTPException(status_code=404, detail=f"Policy model '{model_id}' not found for backend '{backend}'")


def resolve_policy_python_executable(python_executable: str | None = None) -> str:
    if python_executable:
        return python_executable

    candidates = [
        Path(".venv-xvla/bin/python"),
        Path(".venv/bin/python"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.absolute())

    return "python3"


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
    policy: "LivePolicySession | None" = None


@dataclass
class LivePolicySession:
    backend: str
    adapter: Any
    config: dict[str, Any]


@dataclass
class ManagedPolicyServer:
    backend: str
    host: str
    port: int
    model_id: str
    pid: int
    log_path: str
    output_dir: str
    command: list[str]
    process: subprocess.Popen[Any]


@dataclass
class PolicyServerFailure:
    backend: str
    model_id: str
    host: str
    port: int
    pid: int
    exit_code: int | None
    signal: int | None
    log_path: str
    output_dir: str
    command: list[str]
    log_tail: str
    failed_at: float


class SessionManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._session: LiveSession | None = None

    def start(
        self,
        env_name: str,
        task_id: int,
        registry_path: str,
        *,
        extra_camera_names: list[str] | None = None,
    ) -> LiveSession:
        cfg = load_env_config(env_name, registry_path)
        camera_names = list(cfg.get("camera_names", ["agentview"]))
        for camera_name in extra_camera_names or []:
            if camera_name not in camera_names:
                camera_names.append(camera_name)
        env = LiberoExportEnv(
            benchmark_name=cfg["benchmark_name"],
            task_id=task_id,
            camera_names=camera_names,
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
        if session.policy is not None:
            session.policy.adapter.reset()
        return snapshot_state(session.env)

    def attach_policy(self, policy: LivePolicySession) -> LiveSession:
        session = self.get()
        policy.adapter.reset()
        with self._lock:
            session.policy = policy
        return session


class PolicyServerManager:
    def __init__(self) -> None:
        self._lock = Lock()
        self._server: ManagedPolicyServer | None = None
        self._last_failure: PolicyServerFailure | None = None

    def get(self) -> ManagedPolicyServer | None:
        with self._lock:
            server = self._server
        if server is not None and server.process.poll() is not None:
            self._record_failure(server)
            with self._lock:
                self._server = None
            return None
        return server

    def last_failure(self) -> PolicyServerFailure | None:
        with self._lock:
            return self._last_failure

    def _read_log_tail(self, log_path: str) -> str:
        try:
            return Path(log_path).read_text(encoding="utf-8", errors="replace")[-4000:]
        except FileNotFoundError:
            return ""

    def _record_failure(self, server: ManagedPolicyServer) -> None:
        returncode = server.process.poll()
        if returncode is None:
            return

        exit_code = returncode if returncode >= 0 else None
        signal = -returncode if returncode < 0 else None
        failure = PolicyServerFailure(
            backend=server.backend,
            model_id=server.model_id,
            host=server.host,
            port=server.port,
            pid=server.pid,
            exit_code=exit_code,
            signal=signal,
            log_path=server.log_path,
            output_dir=server.output_dir,
            command=server.command,
            log_tail=self._read_log_tail(server.log_path),
            failed_at=time.time(),
        )
        logger.error(
            "Managed policy server exited: backend=%s model=%s pid=%s exit_code=%s signal=%s log=%s",
            failure.backend,
            failure.model_id,
            failure.pid,
            failure.exit_code,
            failure.signal,
            failure.log_path,
        )
        if failure.log_tail:
            logger.error("Managed policy server log tail:\n%s", failure.log_tail)
        with self._lock:
            self._last_failure = failure

    def start(
        self,
        *,
        backend: str,
        model: dict[str, Any],
        host: str,
        port: int,
        device: str,
        python_executable: str | None = None,
    ) -> ManagedPolicyServer:
        existing = self.get()
        if existing is not None:
            raise HTTPException(status_code=409, detail=f"Policy server already running on {existing.host}:{existing.port}")

        if backend != "xvla":
            raise HTTPException(status_code=400, detail=f"Unsupported policy backend '{backend}'")

        python_path = resolve_policy_python_executable(python_executable)
        logger.info(
            "Starting managed policy server: backend=%s model=%s host=%s port=%s device=%s python=%s",
            backend,
            model["id"],
            host,
            port,
            device,
            python_path,
        )

        xvla_root = Path("third_party/policies/xvla").resolve()
        if not xvla_root.exists():
            raise HTTPException(status_code=404, detail=f"XVLA repo not found at {xvla_root}")

        runtime_root = Path("/tmp/sai-inspector-policy-runs").resolve()
        runtime_root.mkdir(parents=True, exist_ok=True)
        run_dir = runtime_root / f"{backend}-{model['id']}-{port}-{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=False)
        log_path = run_dir / "server.log"
        output_dir = run_dir / "runtime"

        command = [
            python_path,
            "deploy.py",
            "--model_path",
            str(model["model_id"]),
            "--processor_path",
            str(model.get("processor_id") or model["model_id"]),
            "--port",
            str(port),
            "--host",
            host,
            "--device",
            device,
            "--output_dir",
            str(output_dir),
            "--disable_slurm",
        ]
        lora_id = model.get("lora_id")
        if lora_id:
            command.extend(["--LoRA_path", str(lora_id)])

        with open(log_path, "wb") as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(xvla_root),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=os.environ.copy(),
            )

        server = ManagedPolicyServer(
            backend=backend,
            host=host,
            port=port,
            model_id=str(model["id"]),
            pid=process.pid,
            log_path=str(log_path),
            output_dir=str(output_dir),
            command=command,
            process=process,
        )
        with self._lock:
            self._server = server
            self._last_failure = None
        logger.info(
            "Managed policy server launched: pid=%s backend=%s model=%s port=%s log=%s",
            server.pid,
            server.backend,
            server.model_id,
            server.port,
            server.log_path,
        )
        return server

    def stop(self) -> dict[str, Any]:
        server = self.get()
        if server is None:
            return {"running": False}

        logger.info(
            "Stopping managed policy server: pid=%s backend=%s model=%s",
            server.pid,
            server.backend,
            server.model_id,
        )
        server.process.terminate()
        try:
            server.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.process.kill()
            server.process.wait(timeout=5)

        with self._lock:
            self._server = None

        return {
            "running": False,
            "pid": server.pid,
        }


class StartSessionRequest(BaseModel):
    env: str
    task_id: int
    registry: str = "src/sai_inspector/registry/libero.yaml"
    policy: "PolicyConfigRequest | None" = None


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


class PolicyConfigRequest(BaseModel):
    backend: str
    model_id: str | None = None
    host: str = "127.0.0.1"
    port: int
    timeout_s: float = 30.0
    restart_if_needed: bool = True


class PolicyServerStartRequest(BaseModel):
    backend: str
    model_id: str | None = None
    host: str = "127.0.0.1"
    port: int = 8001
    device: str = "cuda"
    python_executable: str | None = None


class RolloutStepRequest(BaseModel):
    instruction: str | None = None
    num_steps: int = Field(default=1, ge=1, le=100)


def policy_required_cameras(config: PolicyConfigRequest) -> list[str]:
    if config.backend == "xvla":
        return ["agentview", "robot0_eye_in_hand"]
    raise HTTPException(status_code=400, detail=f"Unsupported policy backend '{config.backend}'")


def build_policy_session(config: PolicyConfigRequest) -> LivePolicySession:
    if config.backend != "xvla":
        raise HTTPException(status_code=400, detail=f"Unsupported policy backend '{config.backend}'")

    model = resolve_policy_model(config.backend, config.model_id)
    adapter = XVLAPolicyAdapter(
        XVLAClient(
            host=config.host,
            port=config.port,
            timeout_s=config.timeout_s,
        )
    )
    return LivePolicySession(
        backend=config.backend,
        adapter=adapter,
        config={**config.model_dump(), "resolved_model": model},
    )


def get_missing_policy_observation_keys(session: LiveSession, config: PolicyConfigRequest) -> list[str]:
    required_cameras = policy_required_cameras(config)
    return [f"{camera_name}_image" for camera_name in required_cameras if f"{camera_name}_image" not in session.env.observation]


def ensure_policy_session_compatibility(config: PolicyConfigRequest) -> LiveSession:
    session = sessions.get()
    missing_keys = get_missing_policy_observation_keys(session, config)
    if not missing_keys:
        return session

    if not config.restart_if_needed:
        missing = ", ".join(missing_keys)
        raise HTTPException(
            status_code=400,
            detail=(
                "Active session is missing policy-required observation keys: "
                f"{missing}. Restart the session with policy enabled or set restart_if_needed=true."
            ),
        )

    return sessions.start(
        session.env_name,
        session.task_id,
        session.registry_path,
        extra_camera_names=policy_required_cameras(config),
    )


app = FastAPI(title="SAI Inspector Backend")
sessions = SessionManager()
policy_servers = PolicyServerManager()


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/policies/catalog")
def get_policy_catalog() -> dict[str, Any]:
    return {"policies": load_policy_catalog()}


@app.post("/api/session/start")
def start_session(req: StartSessionRequest) -> dict[str, Any]:
    registry_path = str(Path(req.registry))
    try:
        extra_camera_names = policy_required_cameras(req.policy) if req.policy is not None else None
        session = sessions.start(
            req.env,
            req.task_id,
            registry_path,
            extra_camera_names=extra_camera_names,
        )
        if req.policy is not None:
            session = sessions.attach_policy(build_policy_session(req.policy))
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
        "policy": session.policy.config if session.policy is not None else None,
    }


@app.get("/api/session")
def get_session() -> dict[str, Any]:
    session = sessions.get()
    return {
        "env_name": session.env_name,
        "task_id": session.task_id,
        "instruction": session.env.instruction,
        "policy": session.policy.config if session.policy is not None else None,
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


@app.post("/api/policy/configure")
def configure_policy(req: PolicyConfigRequest) -> dict[str, Any]:
    ensure_policy_session_compatibility(req)
    session = sessions.attach_policy(build_policy_session(req))
    return {
        "policy": session.policy.config if session.policy is not None else None,
    }


@app.get("/api/policy")
def get_policy() -> dict[str, Any]:
    session = sessions.get()
    return {
        "policy": session.policy.config if session.policy is not None else None,
    }


@app.get("/api/policy/server")
def get_policy_server() -> dict[str, Any]:
    server = policy_servers.get()
    if server is None:
        failure = policy_servers.last_failure()
        if failure is None:
            return {"running": False}
        return {
            "running": False,
            "last_failure": {
                "backend": failure.backend,
                "model_id": failure.model_id,
                "host": failure.host,
                "port": failure.port,
                "pid": failure.pid,
                "exit_code": failure.exit_code,
                "signal": failure.signal,
                "log_path": failure.log_path,
                "output_dir": failure.output_dir,
                "command": failure.command,
                "log_tail": failure.log_tail,
                "failed_at": failure.failed_at,
            },
        }

    healthy = False
    try:
        with socket.create_connection((server.host, server.port), timeout=0.25):
            healthy = True
    except OSError:
        healthy = False

    log_tail = ""
    try:
        log_tail = Path(server.log_path).read_text(encoding="utf-8", errors="replace")[-4000:]
    except FileNotFoundError:
        log_tail = ""

    return {
        "running": server.process.poll() is None,
        "healthy": healthy,
        "backend": server.backend,
        "model_id": server.model_id,
        "host": server.host,
        "port": server.port,
        "pid": server.pid,
        "log_path": server.log_path,
        "log_tail": log_tail,
        "command": server.command,
    }


@app.post("/api/policy/server/start")
def start_policy_server(req: PolicyServerStartRequest) -> dict[str, Any]:
    model = resolve_policy_model(req.backend, req.model_id)
    logger.info(
        "API request start_policy_server: backend=%s model=%s host=%s port=%s device=%s",
        req.backend,
        model["id"],
        req.host,
        req.port,
        req.device,
    )
    server = policy_servers.start(
        backend=req.backend,
        model=model,
        host=req.host,
        port=req.port,
        device=req.device,
        python_executable=req.python_executable,
    )
    return {
        "running": True,
        "backend": server.backend,
        "model_id": server.model_id,
        "host": server.host,
        "port": server.port,
        "pid": server.pid,
        "log_path": server.log_path,
    }


@app.post("/api/policy/server/stop")
def stop_policy_server() -> dict[str, Any]:
    return policy_servers.stop()


@app.post("/api/rollout/step")
def rollout_step(req: RolloutStepRequest) -> dict[str, Any]:
    session = sessions.get()
    policy = session.policy
    if policy is None:
        raise HTTPException(status_code=404, detail="No active policy configured")

    instruction = req.instruction or session.env.instruction
    steps: list[dict[str, Any]] = []
    done = False

    for _ in range(req.num_steps):
        action = policy.adapter.act(session.env, session.env.observation, instruction)
        _, reward, done, info = session.env.step(action)
        steps.append(
            {
                "action": np.asarray(action, dtype=np.float32).tolist(),
                "reward": float(reward),
                "done": bool(done),
                "info": dict(info),
            }
        )
        if done:
            break

    return {
        "instruction": instruction,
        "steps": steps,
        "state": snapshot_state(session.env),
        "done": done,
    }


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
