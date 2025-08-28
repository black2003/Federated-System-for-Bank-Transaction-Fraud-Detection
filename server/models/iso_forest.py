# server/models/iso_forest.py
import numpy as np
from sklearn.ensemble import IsolationForest

class IsoForestWrapper:
    def __init__(self, n_estimators:int=100, contamination:float=0.01, random_state:int=42):
        self.model = IsolationForest(n_estimators=n_estimators,
                                     contamination=contamination,
                                     random_state=random_state,
                                     warm_start=True)
        self._trained = False

    def fit(self, X: np.ndarray):
        if X is None or len(X)==0:
            return
        self.model.fit(X)
        self._trained = True

    def partial_refit(self, X_new: np.ndarray):
        # simulate by re-fitting on X_new (caller should supply sliding buffer)
        self.fit(X_new)

    def anomaly_score(self, X: np.ndarray):
        if X is None or len(X)==0:
            return np.zeros((0,), dtype=float)
        if not self._trained:
            # neutral score (low anomaly)
            return np.zeros((X.shape[0],), dtype=float)
        scores = -self.model.score_samples(X)   # higher -> more anomalous
        smin, smax = scores.min(), scores.max()
        if smax > smin:
            return (scores - smin) / (smax - smin)
        return np.zeros_like(scores)
