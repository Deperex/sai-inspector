from pathlib import Path
import json
import numpy as np
import mujoco
import mujoco.viewer

bundle_dir = Path("/home/getent/repos/sai-inspector/exports/libero-spatial-v0/task_0")
xml_path = bundle_dir / "scene.xml"
state_path = bundle_dir / "state.json"

with open(state_path, "r") as f:
    state = json.load(f)

model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

qpos = np.asarray(state["qpos"], dtype=np.float64)
qvel = np.asarray(state["qvel"], dtype=np.float64)

print("model.nq =", model.nq, "len(qpos) =", len(qpos))
print("model.nv =", model.nv, "len(qvel) =", len(qvel))

assert len(qpos) == model.nq, f"qpos length mismatch: {len(qpos)} vs {model.nq}"
assert len(qvel) == model.nv, f"qvel length mismatch: {len(qvel)} vs {model.nv}"

data.qpos[:] = qpos
data.qvel[:] = qvel

mujoco.mj_forward(model, data)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()