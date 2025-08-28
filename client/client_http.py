# client/client_http.py
import requests
import json
import threading
import time
import numpy as np
import io
import torch

SERVER = "http://127.0.0.1:8000"

class ClientHTTP:
    def __init__(self, client_id, password):
        self.client_id = client_id
        self.password = password
        self.token = None
        self.buffer = []
        self.last_pred = None

    def login(self):
        r = requests.post(f"{SERVER}/login", data={"client_id": self.client_id, "password": self.password})
        if r.status_code != 200:
            raise Exception("Login failed: " + r.text)
        j = r.json()
        self.token = j["token"]
        print("Logged in, token:", self.token)

    def start_stream_listener(self):
        import websocket  # websocket-client package would be required, but we will show a simple fetch loop instead
        # For simplicity, we'll poll the server by calling /predict on pulled synthetic batches
        def poll_loop():
            step = 0
            while True:
                # For demo, fetch a synthetic batch by asking server to predict an empty batch is not available;
                # Instead we'll fetch directly from a local helper: this client assumes ServerApp exposes an endpoint to generate batch.
                time.sleep(1.0)
        t = threading.Thread(target=poll_loop, daemon=True)
        t.start()

    def fetch_batch_via_predict(self, n=10):
        # This uses the server's synthetic data by calling predict with a generated batch payload
        # In practice the client would receive transactions via WebSocket.
        # For demo, we'll generate transactions locally in a shape matching the server and then call /predict
        # BUT the canonical way is to connect websocket; to keep dependencies minimal we'll fetch local synthetic via /predict by sending our own fake batch
        batch = []
        # Ask WebSocket would be ideal; here we'll just craft a small random batch similar to the server
        from server.data.stream import synth_txn
        for i in range(n):
            batch.append(synth_txn(i+1))
        r = requests.post(f"{SERVER}/predict", data={"token": self.token, "batch": json.dumps(batch)})
        if r.status_code != 200:
            raise Exception("Predict failed: " + r.text)
        j = r.json()
        self.last_pred = j
        print("Predicted:", j["p_fused"][:5])
        # store transactions in buffer
        self.buffer.extend(batch)
        return batch, j

    def local_retrain_and_push(self):
        # For demo, we will create a local LSTM, train on labeled buffer where isFraud available
        from server.models.lstm_model import LSTMClassifier
        X = []
        y = []
        for tx in self.buffer:
            X.append(self._featurize_local(tx))
            y.append(tx["isFraud"])
        X = np.vstack(X) if X else np.zeros((0, 12))
        y = np.array(y, dtype=float)
        mask = y != -1
        if mask.sum() == 0:
            print("No labeled data to train.")
            return
        Xl = X[mask]
        yl = y[mask]
        clf = LSTMClassifier(input_dim=X.shape[1])
        clf.train_one_epoch(Xl, yl, batch_size=32)
        # package state dict to bytes and push
        bio = io.BytesIO()
        torch.save(clf.state_dict(), bio)
        bio.seek(0)
        files = {"lstm_state": ("state.pt", bio, "application/octet-stream")}
        # create if snapshot as npy
        buf = np.vstack([self._featurize_local(tx) for tx in self.buffer]) if self.buffer else np.zeros((0, X.shape[1]))
        buf_bio = io.BytesIO()
        np.save(buf_bio, buf)
        buf_bio.seek(0)
        files["if_snapshot"] = ("if.npy", buf_bio, "application/octet-stream")
        r = requests.post(f"{SERVER}/push_update", data={"token": self.token}, files=files)
        print("Push update response:", r.status_code, r.text)

    def _featurize_local(self, tx):
        # same as server.featurize (simple copy)
        delta_org = tx["oldbalanceOrg"] - tx["newbalanceOrig"]
        delta_dest = tx["newbalanceDest"] - tx["oldbalanceDest"]
        base = [tx["amount"], tx["oldbalanceOrg"], tx["newbalanceOrig"], tx["oldbalanceDest"], tx["newbalanceDest"], delta_org, delta_dest]
        TYPE_MAP = {"PAYMENT":0, "TRANSFER":1, "CASH_OUT":2, "DEBIT":3, "CASH_IN":4}
        one_hot = [0]*5
        one_hot[TYPE_MAP.get(tx["type"], 0)] = 1
        return np.array(base + one_hot, dtype=float)

if __name__ == "__main__":
    c = ClientHTTP("client1", "password1")
    c.login()
    batch, pred = c.fetch_batch_via_predict(n=20)
    c.local_retrain_and_push()
