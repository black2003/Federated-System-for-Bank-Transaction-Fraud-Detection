import os
from typing import Dict, Any, List
import numpy as np
import threading

from .utils.serialization import MODELS_DIR, save_torch, load_torch, save_joblib, load_joblib
from .models.lstm_model import LSTMClassifier
from .models.iso_forest import IsoForestWrapper
from .models.fusion import Fusion

GLOBAL_LSTM = os.path.join(MODELS_DIR, "global_lstm.pt")
GLOBAL_IF_JOBLIB = os.path.join(MODELS_DIR, "global_if.joblib")

class GlobalModel:
    def __init__(self, input_dim: int):
        self.lock = threading.Lock()        # protect updates across HTTP requests
        self.lstm = LSTMClassifier(input_dim=input_dim)
        self.iso = IsoForestWrapper()
        self.fusion = Fusion()
        self.input_dim = input_dim
        # Try loading persisted LSTM and IF
        self.load()

    def predict(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        # no lock for read-only predictions (LSTM predict_proba is threadsafe enough)
        p_lstm = self.lstm.predict_proba(X)
        s_iso = self.iso.anomaly_score(X)
        p_fused = self.fusion.fuse(p_lstm, s_iso)
        return {"p_lstm": p_lstm, "s_iso": s_iso, "p_fused": p_fused}

    def aggregate_lstm(self, client_states: List[Dict[str, Any]]):
        # FedAvg over state_dicts (simple arithmetic mean)
        if not client_states:
            return
        with self.lock:
            avg_sd = None
            for sd in client_states:
                if avg_sd is None:
                    avg_sd = {k: v.clone().detach() for k, v in sd.items()}
                else:
                    for k in avg_sd:
                        avg_sd[k] += sd[k]
            for k in avg_sd:
                avg_sd[k] /= float(len(client_states))
            self.lstm.load_state_dict(avg_sd)

    def update_iso(self, X_buffer: np.ndarray | None):
        if X_buffer is None or len(X_buffer) == 0:
            return
        with self.lock:
            # Refit Isolation Forest on the provided buffer snapshot
            self.iso.partial_refit(X_buffer)
            # persist IF to disk
            save_joblib(GLOBAL_IF_JOBLIB, self.iso.model)

    def save(self):
        with self.lock:
            save_torch(GLOBAL_LSTM, self.lstm.state_dict())
            # persist IF model as well
            try:
                save_joblib(GLOBAL_IF_JOBLIB, self.iso.model)
            except Exception:
                pass

    def load(self):
        # load LSTM
        sd = load_torch(GLOBAL_LSTM)
        if sd is not None:
            try:
                self.lstm.load_state_dict(sd)
            except Exception:
                pass

        # load IF via joblib and wrap it again
        model = load_joblib(GLOBAL_IF_JOBLIB)
        if model is not None:
            # Attach the loaded model into wrapper
            try:
                self.iso.model = model
                self.iso._trained = True
            except Exception:
                # if incompatible, ignore
                pass
