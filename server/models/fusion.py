import numpy as np
from sklearn.ensemble import IsolationForest

class Fusion:
    def __init__(self, w_lstm: float = 0.7, w_iso: float = 0.3):
        self.w_lstm = float(w_lstm)
        self.w_iso = float(w_iso)

    def fuse(self, p_lstm: np.ndarray, s_iso: np.ndarray) -> np.ndarray:
        if p_lstm is None or s_iso is None:
            return np.zeros((0,), dtype=float)
        p = np.asarray(p_lstm, dtype=float).reshape(-1)
        s = np.asarray(s_iso, dtype=float).reshape(-1)
        n = min(p.shape[0], s.shape[0])
        if n == 0:
            return np.zeros((0,), dtype=float)
        p = p[:n]
        s = s[:n]
        fused = self.w_lstm * p + self.w_iso * s
        # clip to [0,1]
        return np.clip(fused, 0.0, 1.0)

# Legacy demo model retained for backward compatibility
class FusionModel:
    def __init__(self):
        self.lstm_model = None  # Placeholder for deep model
        self.iforest = IsolationForest(n_estimators=50, contamination=0.01)
        self.is_trained = False

    def train_local(self):
        X = np.random.rand(100, 5)
        self.iforest.fit(X)
        self.is_trained = True
        return {"samples": 100, "features": 5}

    def predict(self, x):
        if not self.is_trained:
            return {"error": "Model not trained"}
        return {"fraud": int(self.iforest.predict([x])[0] == -1)}

    def aggregate_from_clients(self, clients):
        for cid, client in clients.items():
            if client["model"].is_trained:
                self.iforest = client["model"].iforest
                self.is_trained = True
                break
