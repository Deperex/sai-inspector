# sai-inspector

## Clone and setup

```bash
git clone --recurse-submodules https://github.com/Deperex/sai-inspector.git
cd sai-inspector
bash scripts/setup.sh
```

## Run the inspector

The simplest way to export a LIBERO task and start the frontend is:

```bash
bash scripts/run_inspector.sh libero-spatial-v0 0
```

This will:
- export the requested task into `exports/`
- start the Vite frontend on port `5173`

Then open:

```text
http://localhost:5173/?env=libero-spatial-v0&task=0
```

## Run the steps manually

If you want to export first and run the frontend separately:

```bash
source .venv/bin/activate
python scripts/export_libero_env.py --env libero-spatial-v0 --task-id 0
cd frontend
pnpm dev
```

Then open:

```text
http://localhost:5173
```

or load the exported manifest directly:
if you have loaded other export tasks to switch between tasks change the task_0 in the bottom line to task_1 and so on

```text
/exports/libero-spatial-v0/task_0/manifest.json
```
