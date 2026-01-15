from pathlib import Path
import yaml


PRODUCTION_CONFIG_PATH = Path("configs/production.yaml")


def load_production_config() -> dict:
    """
    Load the frozen production configuration.
    Fail fast if it does not exist.
    """
    if not PRODUCTION_CONFIG_PATH.exists():
        raise RuntimeError(
            "Production config not found. Expected configs/production.yaml"
        )

    with PRODUCTION_CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)