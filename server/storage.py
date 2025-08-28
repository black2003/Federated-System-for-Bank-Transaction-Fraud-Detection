# server/storage.py
import os
import json
import time
from typing import Dict, Any, Optional
import joblib
import torch

BASE_DIR = os.path.join(os.getcwd(), "storage")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

CLIENTS_FILE = os.path.join(BASE_DIR, "clients.json")
PREDICTIONS_FILE = os.path.join(BASE_DIR, "predictions.json")
LSTM_PATH = os.path.join(MODELS_DIR, "global_lstm.pt")
IF_PATH = os.path.join(MODELS_DIR, "global_if.joblib")
REGISTRY_DEFAULT = {}

def _safe_read_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def _safe_write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

class Storage:
    def __init__(self):
        os.makedirs(BASE_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
        # init files
        if not os.path.exists(CLIENTS_FILE):
            _safe_write_json(CLIENTS_FILE, {})
        if not os.path.exists(PREDICTIONS_FILE):
            _safe_write_json(PREDICTIONS_FILE, [])

    # Clients (simple username/password store for demo; replace with hashed in prod)
    def add_client(self, cid: str, name: str, password: str) -> bool:
        clients = _safe_read_json(CLIENTS_FILE, {})
        if cid in clients:
            return False
        clients[cid] = {"name": name, "password": password, "last_seen": time.time()}
        _safe_write_json(CLIENTS_FILE, clients)
        return True

    def remove_client(self, cid: str) -> bool:
        clients = _safe_read_json(CLIENTS_FILE, {})
        if cid not in clients:
            return False
        del clients[cid]
        _safe_write_json(CLIENTS_FILE, clients)
        return True

    def get_all_clients(self) -> Dict[str, Any]:
        return _safe_read_json(CLIENTS_FILE, {})

    def touch_client(self, cid: str):
        clients = _safe_read_json(CLIENTS_FILE, {})
        if cid in clients:
            clients[cid]["last_seen"] = time.time()
            _safe_write_json(CLIENTS_FILE, clients)

    # Predictions
    def save_prediction(self, tx: dict, pred: float):
        arr = _safe_read_json(PREDICTIONS_FILE, [])
        arr.append({"time": time.time(), "tx": tx, "pred": float(pred)})
        # keep last 200 for dashboard
        arr = arr[-200:]
        _safe_write_json(PREDICTIONS_FILE, arr)

    def get_recent_predictions(self):
        return _safe_read_json(PREDICTIONS_FILE, [])

    # Model persistence (LSTM - torch, IF - joblib)
    def save_lstm(self, state_dict):
        torch.save(state_dict, LSTM_PATH)

    def load_lstm(self):
        if not os.path.exists(LSTM_PATH):
            return None
        try:
            return torch.load(LSTM_PATH, map_location="cpu")
        except Exception:
            return None

    def save_if(self, iso_model):
        try:
            joblib.dump(iso_model, IF_PATH)
            return True
        except Exception:
            return False

    def load_if(self):
        if not os.path.exists(IF_PATH):
            return None
        try:
            return joblib.load(IF_PATH)
        except Exception:
            return None
