from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import numpy as np

PayloadT = TypeVar("PayloadT")
RawActionT = TypeVar("RawActionT")


class PolicyClient(ABC, Generic[PayloadT, RawActionT]):
    """Transport for sending model-ready payloads to a policy backend."""

    @abstractmethod
    def infer(self, payload: PayloadT) -> RawActionT:
        """Run inference against the backing policy service."""


class ObsPreprocessor(ABC, Generic[PayloadT]):
    """Builds a backend-specific payload from environment observations."""

    def reset(self) -> None:
        """Reset any episode-local state."""

    @abstractmethod
    def transform(self, env: Any, obs: dict[str, Any], instruction: str) -> PayloadT:
        """Convert the current observation into an inference payload."""


class ActionPostprocessor(ABC, Generic[RawActionT]):
    """Consumes raw backend outputs and exposes environment actions."""

    def reset(self) -> None:
        """Reset any episode-local state."""

    @property
    @abstractmethod
    def has_action(self) -> bool:
        """Whether a ready-to-emit environment action is queued."""

    @abstractmethod
    def update_plan(self, raw_action: RawActionT) -> None:
        """Load one or more environment actions from a raw policy output."""

    @abstractmethod
    def pop_action(self) -> np.ndarray:
        """Return the next environment action."""


class PolicyAdapter(Generic[PayloadT, RawActionT]):
    """Generic orchestrator for preprocess -> infer -> postprocess policy flows."""

    def __init__(
        self,
        client: PolicyClient[PayloadT, RawActionT],
        obs_preprocessor: ObsPreprocessor[PayloadT],
        action_postprocessor: ActionPostprocessor[RawActionT],
    ) -> None:
        self.client = client
        self.obs_preprocessor = obs_preprocessor
        self.action_postprocessor = action_postprocessor

    def reset(self) -> None:
        self.obs_preprocessor.reset()
        self.action_postprocessor.reset()

    def on_raw_action(self, raw_action: RawActionT) -> None:
        """Hook for policy-specific state updates after inference."""

    def act(self, env: Any, obs: dict[str, Any], instruction: str) -> np.ndarray:
        if not self.action_postprocessor.has_action:
            payload = self.obs_preprocessor.transform(env, obs, instruction)
            raw_action = self.client.infer(payload)
            self.on_raw_action(raw_action)
            self.action_postprocessor.update_plan(raw_action)

        return self.action_postprocessor.pop_action()
