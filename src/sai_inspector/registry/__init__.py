from pathlib import Path
import yaml


def load_registry(registry_path: str | None = None) -> dict:
    if registry_path is None:
        registry_path = Path(__file__).with_name("libero.yaml")
    else:
        registry_path = Path(registry_path)

    with open(registry_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_env_config(env_name: str, registry_path: str | None = None) -> dict:
    data = load_registry(registry_path)
    envs = data.get("envs", {})
    if env_name not in envs:
        raise KeyError(f"Env '{env_name}' not found in registry")
    return envs[env_name]