from __future__ import annotations

import os
from pathlib import Path
from typing import Any

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

        benchmark_ctor = get_benchmark_dict()[benchmark_name]
        self.benchmark = benchmark_ctor(task_order_index=self.task_order_index)
        self.task = self.benchmark.get_task(self.task_id)

        # Resolve paths relative to THIS repo, not LIBERO's stale internal config.
        repo_root = Path(__file__).resolve().parents[3]
        libero_root = repo_root / "libero" / "libero" / "libero"

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
        return self._control.reset()

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