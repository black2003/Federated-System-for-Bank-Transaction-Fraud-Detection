# server/federated_manager.py
import threading
import numpy as np
from server.models.lstm_model import LSTMClassifier
from server.models.iso_forest import IsoForestWrapper
from server.storage import Storage

class FederatedManager:
    def __init__(self, storage: Storage, input_dim:int=12):
        self.lock = threading.Lock()
        self.storage = storage
        self.input_dim = input_dim
        # instantiate models
        self.lstm = LSTMClassifier(input_dim=input_dim)
        self.iso = IsoForestWrapper()
        # try load persisted LSTM
        self._load_models()

    def _load_models(self):
        # LSTM
        sd = self.storage.load_lstm()
        if sd is not None:
            try:
                self.lstm.load_state_dict(sd)
                print("[FederatedManager] LSTM state loaded from disk.")
            except Exception as e:
                print("[FederatedManager] Could not load LSTM:", e)
        # IsolationForest
        model = self.storage.load_if()
        if model is not None:
            try:
                self.iso.model = model
                self.iso._trained = True
                print("[FederatedManager] IsolationForest loaded from disk.")
            except Exception as e:
                print("[FederatedManager] Could not load IF:", e)

    def predict(self, tx_dict):
        """
        tx_dict: transaction dict (single)
        returns fused probability
        """
        X = self._featurize(tx_dict).reshape(1, -1)
        p_lstm = self.lstm.predict_proba(X)  # 1-d array
        s_iso = self.iso.anomaly_score(X)
        # fusion weights: change if needed
        w_l, w_i = 0.7, 0.3
        p = w_l * p_lstm + w_i * s_iso
        return float(p[0])

    def update_from_client(self, lstm_state_dict=None, if_buffer=None):
        """
        lstm_state_dict: a PyTorch state_dict (already loaded as actual tensors)
        if_buffer: numpy array (snapshot) used to re-fit IF
        """
        with self.lock:
            if lstm_state_dict is not None:
                # simple overwrite / FedAvg could be used with multiple clients
                try:
                    # we accept dict of tensors or numpy -> convert if needed
                    self.lstm.load_state_dict(lstm_state_dict)
                    # persist
                    self.storage.save_lstm(self.lstm.state_dict())
                except Exception as e:
                    print("[FederatedManager] LSTM update error:", e)
            if if_buffer is not None and len(if_buffer) > 0:
                try:
                    self.iso.partial_refit(if_buffer)
                    self.storage.save_if(self.iso.model)
                except Exception as e:
                    print("[FederatedManager] IF update error:", e)

    @staticmethod
    def _featurize(tx: dict):
        # same featurizer as earlier; numeric features + one-hot for type
        delta_org = tx.get("oldbalanceOrg", 0.0) - tx.get("newbalanceOrig", 0.0)
        delta_dest = tx.get("newbalanceDest", 0.0) - tx.get("oldbalanceDest", 0.0)
        base = [
            float(tx.get("amount", 0.0)),
            float(tx.get("oldbalanceOrg", 0.0)),
            float(tx.get("newbalanceOrig", 0.0)),
            float(tx.get("oldbalanceDest", 0.0)),
            float(tx.get("newbalanceDest", 0.0)),
            float(delta_org), float(delta_dest)
        ]
        TYPE_MAP = {"PAYMENT":0, "TRANSFER":1, "CASH_OUT":2, "DEBIT":3, "CASH_IN":4}
        one_hot = [0]*5
        ttype = tx.get("type", "PAYMENT")
        one_hot[TYPE_MAP.get(ttype, 0)] = 1
        return np.array(base + one_hot, dtype=float)
