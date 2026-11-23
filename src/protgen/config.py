from pathlib import Path
import yaml

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
CONFIG = yaml.safe_load(open(CONFIG_PATH))
