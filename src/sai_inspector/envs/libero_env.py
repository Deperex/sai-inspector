from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIBERO_REPO = _REPO_ROOT / "third_party" / "envs" / "libero"
if str(_LIBERO_REPO) not in sys.path:
    sys.path.insert(0, str(_LIBERO_REPO))

from libero.libero.benchmark import get_benchmark_dict
from libero.libero.envs import OffScreenRenderEnv


class LiberoExportEnv:
    def __init__(
        self,
        *,
        benchmark_name: str,
        task_id: int,
        task_order_index: int = 0,
        camera_names: list[str] | None = None,
        camera_heights: int = 256,
        camera_widths: int = 256,
        seed: int | None = None,
        init_state_index: int | None = None,
        env_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        self.task_id = int(task_id)
        self.task_order_index = int(task_order_index)
        self.camera_names = list(camera_names) if camera_names is not None else ["agentview"]
        self.camera_heights = int(camera_heights)
        self.camera_widths = int(camera_widths)
        self.seed_value = seed
        self.init_state_index = init_state_index
        self._env_kwargs = dict(env_kwargs or {})
        self._last_obs: dict[str, Any] | None = None
        self._last_reward: float = 0.0
        self._last_done: bool = False
        self._last_info: dict[str, Any] = {}
        self._init_states: np.ndarray | None = None

        benchmark_ctor = get_benchmark_dict()[benchmark_name]
        self.benchmark = benchmark_ctor(task_order_index=self.task_order_index)
        self.task = self.benchmark.get_task(self.task_id)
        try:
            self._init_states = self.benchmark.get_task_init_states(self.task_id)
        except Exception:
            self._init_states = None

        # Resolve paths relative to THIS repo, not LIBERO's stale internal config.
        repo_root = Path(__file__).resolve().parents[3]
        libero_root = repo_root / "third_party" / "envs" / "libero" / "libero" / "libero"

        bddl_root = libero_root / "bddl_files"
        init_root = libero_root / "init_files"

        self.bddl_file_name = str(
            bddl_root / self.task.problem_folder / self.task.bddl_file
        )
        self.init_states_path = str(
            init_root / self.task.problem_folder / self.task.init_states_file
        )

        print("Using BDDL:", self.bddl_file_name)
        print("Using init states:", self.init_states_path)

        if not os.path.exists(self.bddl_file_name):
            raise FileNotFoundError(f"BDDL file not found: {self.bddl_file_name}")

        if not os.path.exists(self.init_states_path):
            print(f"[warning] init states file not found: {self.init_states_path}")

        self._control = OffScreenRenderEnv(
            bddl_file_name=self.bddl_file_name,
            camera_names=self.camera_names,
            camera_heights=self.camera_heights,
            camera_widths=self.camera_widths,
            **self._env_kwargs,
        )

        if self.seed_value is not None:
            self._control.seed(self.seed_value)

        self.env = self._control.env
        self.reset()

    def reset(self):
        obs = self._control.reset()
        obs = self._apply_init_state(obs)
        self._configure_absolute_control()
        self._last_obs = dict(obs)
        self._last_reward = 0.0
        self._last_done = False
        self._last_info = {}
        return obs

    def step(self, action: np.ndarray | list[float] | tuple[float, ...]):
        obs, reward, done, info = self._control.step(np.asarray(action, dtype=np.float32))
        self._last_obs = dict(obs)
        self._last_reward = float(reward)
        self._last_done = bool(done)
        self._last_info = dict(info)
        return obs, reward, done, info

    @property
    def sim(self):
        return self.env.sim

    @property
    def model(self):
        return self.env.sim.model._model

    @property
    def data(self):
        return self.env.sim.data

    @property
    def instruction(self) -> str:
        return str(self.task.language)

    @property
    def observation(self) -> dict[str, Any]:
        if self._last_obs is None:
            raise RuntimeError("Observation is not initialized. Call reset() first.")
        return self._last_obs

    def list_scene_objects(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for kind, mapping in (
            ("object", self.env.objects_dict),
            ("fixture", self.env.fixtures_dict),
        ):
            for object_name, obj in sorted(mapping.items()):
                joint_names = list(getattr(obj, "joints", []) or [])
                entries.append(
                    {
                        "name": object_name,
                        "kind": kind,
                        "joints": joint_names,
                    }
                )
        return entries

    def get_object_joint_names(self, object_name: str) -> list[str]:
        obj = self.env.get_object(object_name)
        if obj is None:
            raise KeyError(f"Unknown object name: {object_name}")
        return list(getattr(obj, "joints", []) or [])

    def set_object_joint(self, object_name: str, joint_name: str, value: float, *, clamp: bool = True) -> float:
        joint_names = self.get_object_joint_names(object_name)
        if joint_name not in joint_names:
            raise ValueError(
                f"Joint '{joint_name}' is not attached to object '{object_name}'. "
                f"Known joints: {joint_names}"
            )
        return self.set_joint_value(joint_name, value, clamp=clamp)

    def set_object_joints(
        self,
        object_name: str,
        joint_values: dict[str, float],
        *,
        clamp: bool = True,
    ) -> dict[str, float]:
        updated: dict[str, float] = {}
        for joint_name, value in joint_values.items():
            updated[joint_name] = self.set_object_joint(
                object_name,
                joint_name,
                value,
                clamp=clamp,
            )
        return updated

    def set_object_pose(
        self,
        object_name: str,
        *,
        pos: list[float] | tuple[float, float, float],
        quat: list[float] | tuple[float, float, float, float],
    ) -> dict[str, list[float]]:
        joint_names = self.get_object_joint_names(object_name)
        if len(joint_names) != 1:
            raise ValueError(
                f"Object '{object_name}' must have exactly one free joint to set pose. "
                f"Found joints: {joint_names}"
            )

        joint_name = joint_names[0]
        joint_id = self._joint_id_from_name(joint_name)
        if self._joint_type_name(joint_id) != "free":
            raise ValueError(
                f"Object '{object_name}' is not free-moving; joint '{joint_name}' "
                f"is type '{self._joint_type_name(joint_id)}'"
            )

        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        dof_addr = int(self.model.jnt_dofadr[joint_id])

        pos_arr = np.asarray(pos, dtype=np.float64)
        quat_arr = np.asarray(quat, dtype=np.float64)
        if pos_arr.shape != (3,):
            raise ValueError(f"Expected pos with 3 values, got shape {pos_arr.shape}")
        if quat_arr.shape != (4,):
            raise ValueError(f"Expected quat with 4 values, got shape {quat_arr.shape}")

        quat_norm = float(np.linalg.norm(quat_arr))
        if quat_norm <= 0:
            raise ValueError("Quaternion norm must be positive")
        quat_arr = quat_arr / quat_norm

        self.data.qpos[qpos_addr : qpos_addr + 3] = pos_arr
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = quat_arr
        self.data.qvel[dof_addr : dof_addr + 6] = 0.0
        self._forward()

        return {
            "joint": joint_name,
            "pos": [float(v) for v in self.data.qpos[qpos_addr : qpos_addr + 3]],
            "quat": [float(v) for v in self.data.qpos[qpos_addr + 3 : qpos_addr + 7]],
        }

    def set_joint_value(self, joint_name: str, value: float, *, clamp: bool = True) -> float:
        joint_id = self._joint_id_from_name(joint_name)
        joint_type = self._joint_type_name(joint_id)
        if joint_type not in {"slide", "hinge"}:
            raise ValueError(
                f"set_joint_value only supports slide / hinge joints, got '{joint_type}' "
                f"for joint '{joint_name}'"
            )

        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        dof_addr = int(self.model.jnt_dofadr[joint_id])
        applied_value = float(value)

        if clamp and bool(self.model.jnt_limited[joint_id]):
            lo, hi = [float(v) for v in self.model.jnt_range[joint_id]]
            applied_value = min(max(applied_value, lo), hi)

        self.data.qpos[qpos_addr] = applied_value
        self.data.qvel[dof_addr] = 0.0
        self._forward()
        return float(self.data.qpos[qpos_addr])

    def get_joint_value(self, joint_name: str) -> float:
        joint_id = self._joint_id_from_name(joint_name)
        qpos_addr = int(self.model.jnt_qposadr[joint_id])
        return float(self.data.qpos[qpos_addr])

    def get_object_state(self, object_name: str) -> dict[str, Any]:
        joint_names = self.get_object_joint_names(object_name)
        if not joint_names:
            return {
                "name": object_name,
                "joints": {},
            }

        joint_state: dict[str, Any] = {}
        for joint_name in joint_names:
            joint_id = self._joint_id_from_name(joint_name)
            joint_type = self._joint_type_name(joint_id)
            qpos_addr = int(self.model.jnt_qposadr[joint_id])
            if joint_type == "free":
                joint_state[joint_name] = {
                    "type": joint_type,
                    "pos": [float(v) for v in self.data.qpos[qpos_addr : qpos_addr + 3]],
                    "quat": [float(v) for v in self.data.qpos[qpos_addr + 3 : qpos_addr + 7]],
                }
            else:
                joint_state[joint_name] = {
                    "type": joint_type,
                    "value": float(self.data.qpos[qpos_addr]),
                }

        return {
            "name": object_name,
            "joints": joint_state,
        }

    def _joint_id_from_name(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise KeyError(f"Unknown joint name: {joint_name}")
        return int(joint_id)

    def _joint_type_name(self, joint_id: int) -> str:
        return {
            int(mujoco.mjtJoint.mjJNT_FREE): "free",
            int(mujoco.mjtJoint.mjJNT_BALL): "ball",
            int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
            int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
        }.get(int(self.model.jnt_type[joint_id]), f"unknown_{int(self.model.jnt_type[joint_id])}")

    def _forward(self) -> None:
        self.sim.forward()

    def _apply_init_state(self, obs: dict[str, Any]) -> dict[str, Any]:
        if self._init_states is None or len(self._init_states) == 0:
            return obs

        init_state_index = self.init_state_index if self.init_state_index is not None else 0
        init_state_index = int(init_state_index) % int(self._init_states.shape[0])
        return self._control.set_init_state(self._init_states[init_state_index])

    def _configure_absolute_control(self) -> None:
        for robot in self.env.robots:
            if hasattr(robot.controller, "use_delta"):
                robot.controller.use_delta = False
