from __future__ import annotations

from typing import Any

import numpy as np

from ..base import PolicyAdapter
from .client import XVLAClient
from .libero import XVLAActionPostprocessor, XVLAObsPreprocessor


class XVLAPolicyAdapter(PolicyAdapter[dict[str, Any], np.ndarray]):
    """Composes XVLA preprocessing, transport, and action postprocessing."""

    def __init__(
        self,
        client: XVLAClient,
        *,
        obs_preprocessor: XVLAObsPreprocessor | None = None,
        action_postprocessor: XVLAActionPostprocessor | None = None,
    ) -> None:
        super().__init__(
            client=client,
            obs_preprocessor=obs_preprocessor or XVLAObsPreprocessor(),
            action_postprocessor=action_postprocessor or XVLAActionPostprocessor(),
        )

    def on_raw_action(self, raw_action: np.ndarray) -> None:
        self.obs_preprocessor.update_proprio_from_xvla_plan(raw_action)
