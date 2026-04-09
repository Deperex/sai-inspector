from __future__ import annotations

from typing import Any

import numpy as np
import requests

from ..base import PolicyClient


class XVLAClient(PolicyClient[dict[str, Any], np.ndarray]):
    """Thin HTTP client for an XVLA inference server."""

    def __init__(self, host: str, port: int, *, timeout_s: float = 30.0) -> None:
        self.url = f"http://{host}:{port}/act"
        self.timeout_s = float(timeout_s)

    def infer(self, payload: dict[str, Any]) -> np.ndarray:
        try:
            response = requests.post(self.url, json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            raise RuntimeError(f"XVLA request failed: {exc}") from exc

        action = np.asarray(data["action"])
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected XVLA action shape from server: {action.shape}")
        return action
