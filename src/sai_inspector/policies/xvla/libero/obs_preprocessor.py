from __future__ import annotations

from typing import Any

import numpy as np

from ...base import ObsPreprocessor
from .action_postprocessor import LiberoAbsActionProcessor

try:
    import json_numpy
except ModuleNotFoundError:
    json_numpy = None


def flip_agentview(image: np.ndarray) -> np.ndarray:
    """Match the original XVLA client convention for agentview images."""
    return np.flip(np.flip(image, 0), 1)


def dumps_array(value: np.ndarray) -> Any:
    if json_numpy is not None:
        return json_numpy.dumps(value)
    return value.tolist()


class XVLAObsPreprocessor(ObsPreprocessor[dict[str, Any]]):
    """Formats LIBERO observations into the payload XVLA expects."""

    def __init__(self, *, domain_id: int = 3, plan_steps: int = 10) -> None:
        self.domain_id = int(domain_id)
        self.plan_steps = int(plan_steps)
        self.processor = LiberoAbsActionProcessor()
        self.proprio: np.ndarray | None = None

    def reset(self) -> None:
        self.proprio = None

    def _extract_robot_state(self, env: Any, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        robot = env.env.robots[0]
        robo_ori = self.processor.mat_to_rotate6d(robot.controller.ee_ori_mat)
        robo_pos = robot.controller.ee_pos
        obs["robo_ori"] = robo_ori
        obs["robo_pos"] = robo_pos
        return robo_pos, robo_ori

    def transform(self, env: Any, obs: dict[str, Any], instruction: str) -> dict[str, Any]:
        robo_pos, robo_ori = self._extract_robot_state(env, obs)

        closed_loop_proprio = np.concatenate(
            [robo_pos, robo_ori, np.array([0.0], dtype=np.float32)],
            axis=-1,
        )
        closed_loop_proprio = np.concatenate(
            [closed_loop_proprio, np.zeros_like(closed_loop_proprio)],
            axis=-1,
        )

        if self.proprio is None:
            self.proprio = closed_loop_proprio.copy()

        return {
            "proprio": dumps_array(self.proprio),
            "language_instruction": instruction,
            "image0": dumps_array(flip_agentview(obs["agentview_image"])),
            "image1": dumps_array(obs["robot0_eye_in_hand_image"]),
            "domain_id": self.domain_id,
            "steps": self.plan_steps,
        }

    def update_proprio_from_xvla_plan(self, xvla_action: np.ndarray) -> None:
        if self.proprio is None:
            raise RuntimeError("Proprio is not initialized. Call transform() first.")
        self.proprio[:9] = xvla_action[-1, :9].copy()
