from __future__ import annotations

from collections import deque
from typing import Deque

import numpy as np
import robosuite.utils.transform_utils as T

from ...base import ActionPostprocessor

EPS = 1e-6


class LiberoAbsActionProcessor:
    """Helpers for converting XVLA absolute actions into LIBERO actions."""

    def rotate6d_to_axis_angle(self, r6d: np.ndarray) -> np.ndarray:
        single = False
        if r6d.ndim == 1:
            r6d = r6d[None, :]
            single = True

        a1 = r6d[:, 0:3]
        a2 = r6d[:, 3:6]

        b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + EPS)
        dot_prod = np.sum(b1 * a2, axis=-1, keepdims=True)
        b2_orth = a2 - dot_prod * b1
        b2 = b2_orth / (np.linalg.norm(b2_orth, axis=-1, keepdims=True) + EPS)
        b3 = np.cross(b1, b2, axis=-1)

        rotation = np.stack([b1, b2, b3], axis=-1)
        axis_angle = np.stack(
            [T.quat2axisangle(T.mat2quat(rotation[i])) for i in range(rotation.shape[0])],
            axis=0,
        )
        return axis_angle[0] if single else axis_angle

    def mat_to_rotate6d(self, rotation: np.ndarray) -> np.ndarray:
        if rotation.ndim == 2:
            return np.concatenate([rotation[:3, 0], rotation[:3, 1]], axis=-1)
        if rotation.ndim == 3:
            return np.concatenate([rotation[:, :3, 0], rotation[:, :3, 1]], axis=-1)
        raise ValueError("Rotation matrix must be (..., 3, 3)")

    def action_6d_to_axisangle(self, action: np.ndarray) -> np.ndarray:
        if action.ndim == 1:
            final_ori = self.rotate6d_to_axis_angle(action[3:9])
            return np.concatenate([action[0:3], final_ori, action[-1:]])
        if action.ndim == 2:
            final_ori = self.rotate6d_to_axis_angle(action[:, 3:9])
            return np.concatenate([action[:, 0:3], final_ori, action[:, -1:]], axis=-1)
        raise ValueError("Unsupported action shape")


class XVLAActionPostprocessor(ActionPostprocessor[np.ndarray]):
    """Turns raw XVLA predictions into LIBERO-compatible actions."""

    def __init__(self) -> None:
        self.processor = LiberoAbsActionProcessor()
        self.action_plan: Deque[list[float]] = deque()

    def reset(self) -> None:
        self.action_plan.clear()

    @property
    def has_action(self) -> bool:
        return bool(self.action_plan)

    def update_plan(self, xvla_action: np.ndarray) -> None:
        if xvla_action.ndim != 2 or xvla_action.shape[1] < 10:
            raise ValueError(f"Unexpected XVLA action shape: {xvla_action.shape}")

        target_eef = xvla_action[:, :3]
        target_axis = self.processor.rotate6d_to_axis_angle(xvla_action[:, 3:9])
        target_grip = xvla_action[:, 9:10]
        final_action = np.concatenate([target_eef, target_axis, target_grip], axis=-1)

        for row in final_action.tolist():
            row[-1] = 1.0 if row[-1] > 0.5 else -1.0
            self.action_plan.append(row)

    def pop_action(self) -> np.ndarray:
        if not self.action_plan:
            raise RuntimeError("No action plan available. Call update_plan() first.")
        return np.asarray(self.action_plan.popleft(), dtype=np.float32)
