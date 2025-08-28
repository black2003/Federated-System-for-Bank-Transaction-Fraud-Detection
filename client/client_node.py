# client/client_node.py
import requests, json, io, numpy as np, torch
from server.data_legacy import stream as stream_module  # legacy synthetic generator

SERVER = "http://127.0.0.1:8000"

def register_local(name, pwd):
    r = requests.post(f"{SERVER}/api/register", json={"name": name, "password": pwd})
    print("register:", r.status_code, r.text)

def send_tx_and_get_prediction(tx):
    r = requests.post(f"{SERVER}/api/predict", json=tx)
    print("predict:", r.status_code, r.json())
    return r.json()

def push_update_example(local_clf_state_dict, buffer_array):
    # local_clf_state_dict: a state_dict (torch)
    bio = io.BytesIO()
    torch.save(local_clf_state_dict, bio)
    bio.seek(0)
    npbuf = io.BytesIO()
    np.save(npbuf, buffer_array)
    npbuf.seek(0)
    files = {
        "lstm_state": ("state.pt", bio, "application/octet-stream"),
        "if_snapshot": ("buf.npy", npbuf, "application/octet-stream")
    }
    r = requests.post(f"{SERVER}/api/push_update", files=files)
    print("push_update:", r.status_code, r.text)

if __name__ == "__main__":
    # register demo client
    register_local("demo-bank", "demo-pass")
    # craft a sample transaction
    tx = {
      "step": 1, "type":"TRANSFER", "amount": 12000.0,
      "nameOrig":"C1", "oldbalanceOrg":50000.0, "newbalanceOrig":38000.0,
      "nameDest":"C2", "oldbalanceDest": 1000.0, "newbalanceDest":13000.0,
      "isFraud":0, "isFlaggedFraud":0
    }
    send_tx_and_get_prediction(tx)
