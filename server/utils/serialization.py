import json
import os
from typing import Any, Dict
import torch
import joblib

STORAGE_DIR = os.path.join("storage")
MODELS_DIR = os.path.join(STORAGE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

def save_json(path: str, obj: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_json(path: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if not os.path.exists(path):
        return default or {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_torch(path: str, state_dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state_dict, path)

def load_torch(path: str):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location="cpu")

# New: joblib wrappers for sklearn persistence
def save_joblib(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_joblib(path: str):
    if not os.path.exists(path):
        return None
    return joblib.load(path)
