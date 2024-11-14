import os
import json
from typing import Dict, Any

CONFIG_FILE = os.path.expanduser("~/.aigit/model.json")

def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return {"models": {}, "active_model": None}

def save_config(config: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=2)

def add_model_to_config(name: str, model_config: Dict[str, Any]) -> None:
    config = load_config()
    config["models"][name] = model_config
    if config["active_model"] is None:
        config["active_model"] = name
    save_config(config)

def remove_model_from_config(name: str) -> None:
    config = load_config()
    if name in config["models"]:
        del config["models"][name]
        if config["active_model"] == name:
            config["active_model"] = next(iter(config["models"]), None)
        save_config(config)

def set_active_model_in_config(name: str) -> None:
    config = load_config()
    if name in config["models"]:
        config["active_model"] = name
        save_config(config)

def get_active_model() -> Dict[str, Any]:
    config = load_config()
    active_model = config["active_model"]
    if active_model and active_model in config["models"]:
        return config["models"][active_model]
    return {}
