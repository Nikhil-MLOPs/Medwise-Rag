import yaml
from pathlib import Path

def load_config(base_path: Path, exp_path: Path):
    base = yaml.safe_load(base_path.read_text())
    exp = yaml.safe_load(exp_path.read_text())

    for section, values in exp.items():
        base.setdefault(section, {})
        base[section].update(values)

    return base
