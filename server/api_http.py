# server/api_http.py
import asyncio
import json
import time
from typing import Dict, Any, List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

from .auth import AuthDB
from .federation import GlobalModel
from .data.stream import synth_txn, stream_transactions
from .api import featurize, INPUT_DIM, Registry  # reuse featurize from api module
from .utils.logging_config import setup_logging

logger = setup_logging()

app = FastAPI(title="Fraud Federated Server (HTTP + WS)")
# Allow cross-origin access so the dashboard can point to different server IPs
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
auth = AuthDB()
registry = Registry()
global_model = GlobalModel(input_dim=INPUT_DIM)

# Load the global model
try:
    global_model.load()
    logger.info("Global model loaded successfully")
except Exception as e:
    logger.warning(f"Could not load global model: {e}")
    logger.info("Global model will be initialized with default weights")

# Static and templates for simple Admin/Client consoles
app.mount("/static", StaticFiles(directory="server/static"), name="static")
templates = Jinja2Templates(directory="server/templates")

# Simple in-memory token system for demo (not secure)
SESSIONS: Dict[str, str] = {}  # token -> client_id

def make_token(client_id: str) -> str:
    return f"{client_id}-{int(time.time()*1000)}"

@app.post("/login")
def login(client_id: str = Form(...), password: str = Form(...)):
    if not auth.verify_client(client_id, password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = make_token(client_id)
    SESSIONS[token] = client_id
    return {"token": token, "client_id": client_id}

@app.get("/")
def root_redirect():
    return {"ok": True, "message": "Visit /admin or /client for consoles"}

@app.get("/test")
def test_model():
    """Test endpoint to verify the global model is working."""
    try:
        # Create a simple test transaction
        test_tx = {
            "amount": 1000.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4000.0,
            "oldbalanceDest": 2000.0,
            "newbalanceDest": 3000.0,
            "type": "PAYMENT"
        }
        
        # Test featurization
        features = featurize(test_tx)
        
        # Test prediction
        prediction = global_model.predict(features.reshape(1, -1))
        
        return {
            "ok": True,
            "message": "Model test successful",
            "test_features": features.tolist(),
            "prediction_keys": list(prediction.keys()),
            "model_loaded": True
        }
    except Exception as e:
        import traceback
        return {
            "ok": False,
            "message": f"Model test failed: {e}",
            "traceback": traceback.format_exc(),
            "model_loaded": False
        }

@app.get("/model_status")
def model_status():
    """Check if the global model is working properly."""
    try:
        # Create a simple test transaction
        test_tx = {
            "amount": 1000.0,
            "oldbalanceOrg": 5000.0,
            "newbalanceOrig": 4000.0,
            "oldbalanceDest": 2000.0,
            "newbalanceDest": 3000.0,
            "type": "PAYMENT"
        }
        
        # Test featurization
        features = featurize(test_tx)
        
        # Test prediction
        prediction = global_model.predict(features.reshape(1, -1))
        
        return {
            "status": "OK",
            "message": "Global model is working properly",
            "input_dim": INPUT_DIM,
            "feature_vector_length": len(features),
            "prediction_available": True,
            "prediction_keys": list(prediction.keys())
        }
    except Exception as e:
        import traceback
        return {
            "status": "ERROR",
            "message": f"Global model error: {e}",
            "input_dim": INPUT_DIM,
            "traceback": traceback.format_exc()
        }

@app.get("/admin")
def admin_console(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/client")
def client_console(request: Request):
    return templates.TemplateResponse("client.html", {"request": request})

@app.get("/admin/clients")
def list_clients():
    """Return current registered clients for Admin console."""
    return {"clients": registry.list()}

# ---- ZKP Integration ----
def _zkp_paths():
    import os
    repo_root = os.path.abspath(os.getcwd())
    zkp_root = os.path.join(repo_root, "ZKP", "MitsubishiProd")
    deploy_dir = os.path.join(zkp_root, "deploy")
    return zkp_root, deploy_dir

def _classify_with_zkp(features):
    """Call ZKP classify util with provided feature vector; returns dict with keys prediction, proof_path, zkp_status."""
    import os, sys
    zkp_root, _ = _zkp_paths()
    if zkp_root not in sys.path:
        sys.path.insert(0, zkp_root)
    try:
        from api.zkp_utils import classify_with_zkp
        return classify_with_zkp(features)
    except Exception as e:
        # Fallback: no ZKP environment; return passthrough prediction format
        return {"prediction": 0, "proof_path": "", "zkp_status": f"ZKP unavailable: {e}"}

@app.post("/zkp/export_tx")
async def zkp_export_tx(token: str = Form(...), tx_json: str = Form(...), p_fused: str = Form(None), pred: str = Form(None)):
    """Featurize a transaction, run ZKP classification, and export message to ZKP deploy dir."""
    if token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        import json, os
        tx = json.loads(tx_json)
        feats = featurize(tx)
        # If client already computed prediction, we can pass-through; otherwise run ZKP classify
        res = _classify_with_zkp(feats)
        message = {
            "sample_id": int(tx.get("step", 0)),
            "prediction": int(pred) if pred is not None else int(res.get("prediction", 0)),
            "true_label": tx.get("isFraud", None),
            "proof_path": res.get("proof_path", ""),
            "zkp_status": res.get("zkp_status", "")
        }
        if p_fused is not None:
            try:
                message["p_fused"] = float(p_fused)
            except Exception:
                pass
        _, deploy_dir = _zkp_paths()
        os.makedirs(deploy_dir, exist_ok=True)
        out_path = os.path.join(deploy_dir, "bank_message.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(message, f, indent=2)
        return {"ok": True, "path": out_path, "message": message}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ZKP export failed: {e}")

@app.get("/zkp/status")
def zkp_status():
    """Quick check for ZKP environment and last exported message path."""
    try:
        import os, sys
        zkp_root, deploy_dir = _zkp_paths()
        available = True
        try:
            if zkp_root not in sys.path:
                sys.path.insert(0, zkp_root)
            from api.zkp_utils import classify_with_zkp  # noqa: F401
        except Exception:
            available = False
        msg_path = os.path.join(deploy_dir, "bank_message.json")
        return {"available": available, "deploy_dir": deploy_dir, "message_exists": os.path.exists(msg_path)}
    except Exception as e:
        return {"available": False, "error": str(e)}

@app.post("/admin/bootstrap")
def admin_bootstrap(password: str = Form(...)):
    """One-time endpoint to set admin password non-interactively."""
    if auth.has_admin():
        raise HTTPException(status_code=400, detail="Admin already configured")
    if not auth.set_admin(password):
        raise HTTPException(status_code=400, detail="Password invalid or already set")
    return {"ok": True}

@app.post("/admin/register_client")
def admin_register(admin_password: str = Form(...), client_id: str = Form(...), name: str = Form(...), password: str = Form(...)):
    if not auth.verify_admin(admin_password):
        raise HTTPException(status_code=401, detail="Admin auth failed")
    ok = auth.register_client(client_id, name, password)
    if not ok:
        raise HTTPException(status_code=400, detail="Client already exists")
    registry.add(client_id, name)
    return {"ok": True}

@app.post("/admin/remove_client")
def admin_remove(admin_password: str = Form(...), client_id: str = Form(...)):
    if not auth.verify_admin(admin_password):
        raise HTTPException(status_code=401, detail="Admin auth failed")
    ok = auth.delete_client(client_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Client not found")
    registry.remove(client_id)
    return {"ok": True}

@app.post("/admin/train_global_dataset")
async def admin_train_global_dataset(
    admin_password: str = Form(...),
    dataset: UploadFile = File(...),
    epochs: int = Form(1)
):
    """
    Admin uploads a CSV dataset; server trains/updates the GLOBAL model.
    Expected CSV columns: amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type, isFraud
    NOTE: isFraud column IS required for training (supervised learning needs labels)
    """
    if not auth.verify_admin(admin_password):
        raise HTTPException(status_code=401, detail="Admin auth failed")
    try:
        import io, csv, numpy as np
        content = await dataset.read()
        text = content.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        
        # Check if required columns exist (including isFraud for training)
        if reader.fieldnames:
            required_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type", "isFraud"]
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                raise HTTPException(status_code=400, detail=f"Missing required columns for training: {missing_cols}. Training requires labeled data with 'isFraud' column.")
        
        X_rows = []
        y_rows = []
        row_count = 0
        error_count = 0
        
        for row in reader:
            row_count += 1
            try:
                # cast numeric fields
                for k in [
                    "amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"
                ]:
                    row[k] = float(row.get(k, 0) or 0)
                
                # isFraud is required for training
                if "isFraud" not in row or row["isFraud"] == "":
                    print(f"Row {row_count}: Missing isFraud label (required for training)")
                    error_count += 1
                    continue
                
                try:
                    row["isFraud"] = int(float(row.get("isFraud", 0) or 0))
                except (ValueError, TypeError) as e:
                    print(f"Row {row_count}: Error converting isFraud='{row.get('isFraud')}' to int: {e}")
                    error_count += 1
                    continue
                
                X_rows.append(featurize(row))
                y_rows.append(row["isFraud"])
            except Exception as e:
                print(f"Row {row_count}: Error processing: {e}")
                error_count += 1
                continue
        
        print(f"Training: Processed {row_count} rows, {len(X_rows)} successful, {error_count} errors")
        
        if not X_rows:
            raise HTTPException(status_code=400, detail=f"No usable rows in CSV. Processed {row_count} rows, {error_count} errors. Training requires labeled data with 'isFraud' column.")
        
        X = np.vstack(X_rows)
        y = np.array(y_rows, dtype=float)
        
        # Train LSTM for a few epochs
        for epoch in range(max(1, int(epochs))):
            print(f"Training epoch {epoch + 1}/{epochs}")
            global_model.lstm.train_one_epoch(X, y, batch_size=128)
        
        # Refresh IF using all X (unsupervised)
        print("Updating Isolation Forest...")
        global_model.iso.partial_refit(X)
        
        # Save the updated model
        print("Saving global model...")
        global_model.save()
        
        return {"ok": True, "samples": int(X.shape[0]), "epochs": int(epochs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to train: {e}")

@app.post("/predict")
async def predict(token: str = Form(...), batch: str = Form(...)):
    # batch should be JSON array of transaction dicts (same fields)
    if token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        txs = json.loads(batch)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid batch")
    X = []
    for tx in txs:
        X.append(featurize(tx))
    import numpy as np
    X = np.vstack(X) if X else np.zeros((0, global_model.input_dim))
    out = global_model.predict(X)
    # Return p_fused and p_lstm and s_iso
    return {
        "p_fused": out["p_fused"].tolist(),
        "p_lstm": out["p_lstm"].tolist(),
        "s_iso": out["s_iso"].tolist()
    }

@app.post("/client/evaluate_dataset")
async def client_evaluate_dataset(token: str = Form(...), dataset: UploadFile = File(...)):
    """Client uploads a CSV to evaluate using GLOBAL model; returns predictions.
    Adds any_fraud flag if any fused probability crosses threshold (0.5).
    Note: isFraud column is NOT required - this is what we're predicting!
    If isFraud column exists, it will be ignored (filtered out).
    """
    if token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        import io, csv, numpy as np
        content = await dataset.read()
        text = content.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        
        # Debug: Check column names
        if reader.fieldnames:
            print(f"CSV columns: {reader.fieldnames}")
            
            # Filter out empty columns and isFraud column
            filtered_fieldnames = [col for col in reader.fieldnames if col.strip() and col != 'isFraud']
            print(f"Filtered columns (excluding empty and isFraud): {filtered_fieldnames}")
        
        X_rows = []
        row_count = 0
        error_count = 0
        
        for row in reader:
            row_count += 1
            try:
                # Check if required columns exist (isFraud is NOT required - it's what we're predicting!)
                required_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type"]
                missing_cols = [col for col in required_cols if col not in row]
                if missing_cols:
                    print(f"Row {row_count}: Missing columns: {missing_cols}")
                    error_count += 1
                    continue
                
                # Convert numeric columns
                for k in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
                    try:
                        row[k] = float(row.get(k, 0) or 0)
                    except (ValueError, TypeError) as e:
                        print(f"Row {row_count}: Error converting {k}='{row.get(k)}' to float: {e}")
                        error_count += 1
                        continue
                
                # Check type column
                if "type" not in row or not row["type"]:
                    print(f"Row {row_count}: Missing or empty type column")
                    error_count += 1
                    continue
                
                # Featurize the row
                try:
                    features = featurize(row)
                    X_rows.append(features)
                except Exception as e:
                    print(f"Row {row_count}: Error in featurize: {e}")
                    print(f"Row data: {row}")
                    error_count += 1
                    continue
                    
            except Exception as e:
                print(f"Row {row_count}: Unexpected error: {e}")
                error_count += 1
                continue
        
        print(f"Processed {row_count} rows, {len(X_rows)} successful, {error_count} errors")
        
        if not X_rows:
            raise HTTPException(status_code=400, detail=f"No usable rows in CSV. Processed {row_count} rows, {error_count} errors. Required columns: amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type")
        
        X = np.vstack(X_rows)
        out = global_model.predict(X)
        resp = {
            "p_fused": out["p_fused"].tolist(),
            "p_lstm": out["p_lstm"].tolist(),
            "s_iso": out["s_iso"].tolist()
        }
        
        # Check if any fraud is detected based on threshold
        try:
            import numpy as np
            any_fraud = bool((out["p_fused"] >= 0.5).any())
            resp["any_fraud"] = any_fraud
            print(f"Fraud detection: {any_fraud} (threshold 0.5)")
        except Exception as e:
            print(f"Error checking fraud threshold: {e}")
            resp["any_fraud"] = False
        
        # Note: We don't check accuracy since we don't have ground truth labels
        # The client is predicting on unlabeled data
        
        return resp
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"Evaluation failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail=f"Failed to evaluate: {e}")

@app.post("/push_update")
async def push_update(token: str = Form(...), lstm_state: UploadFile = File(None), if_snapshot: UploadFile = File(None)):
    """
    Accepts:
      - lstm_state: a torch state_dict file (bytes)
      - if_snapshot: a numpy .npy bytes file representing the X buffer (optional)
    For simplicity this demo accepts raw files and applies them immediately.
    """
    if token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not logged in")

    client_id = SESSIONS[token]
    client_states = []
    if lstm_state is not None:
        # read bytes and load into torch
        import io, torch
        content = await lstm_state.read()
        bio = io.BytesIO(content)
        try:
            sd = torch.load(bio, map_location="cpu")
            client_states.append(sd)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid LSTM state dict: {e}")

    if client_states:
        global_model.aggregate_lstm(client_states)
    if if_snapshot is not None:
        content = await if_snapshot.read()
        # assume numpy .npy
        import io, numpy as np
        try:
            buf = io.BytesIO(content)
            arr = np.load(buf, allow_pickle=False)
            global_model.update_iso(arr)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid IF snapshot: {e}")

    global_model.save()
    return {"ok": True}

@app.post("/client/local_retrain_dataset")
async def client_local_retrain_dataset(
    token: str = Form(...),
    dataset: UploadFile = File(...),
    epochs: int = Form(1)
):
    """
    Client uploads labeled CSV -> locally retrain small LSTM and aggregate into GLOBAL.
    Also updates IF using features buffer from CSV.
    Persists a per-client local model copy on first run (initialized from GLOBAL) and
    thereafter updates both the local copy and the GLOBAL model.
    
    NOTE: isFraud column IS required for training (supervised learning needs labels)
    """
    if token not in SESSIONS:
        raise HTTPException(status_code=401, detail="Not logged in")
    try:
        import io, csv, numpy as np, os
        from .models.lstm_model import LSTMClassifier
        from .utils.serialization import MODELS_DIR, save_torch, load_torch
        
        content = await dataset.read()
        text = content.decode("utf-8", errors="ignore")
        reader = csv.DictReader(io.StringIO(text))
        
        # Check if required columns exist (including isFraud for training)
        if reader.fieldnames:
            required_cols = ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "type", "isFraud"]
            missing_cols = [col for col in required_cols if col not in reader.fieldnames]
            if missing_cols:
                raise HTTPException(status_code=400, detail=f"Missing required columns for training: {missing_cols}. Training requires labeled data with 'isFraud' column.")
        
        X_rows = []
        y_rows = []
        row_count = 0
        error_count = 0
        
        for row in reader:
            row_count += 1
            try:
                for k in [
                    "amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"
                ]:
                    row[k] = float(row.get(k, 0) or 0)
                
                # isFraud is required for training
                y = row.get("isFraud", "")
                if y == "":
                    print(f"Row {row_count}: Missing isFraud label (required for training)")
                    error_count += 1
                    continue
                
                try:
                    y_val = int(float(y))
                    y_rows.append(y_val)
                except (ValueError, TypeError) as e:
                    print(f"Row {row_count}: Error converting isFraud='{y}' to int: {e}")
                    error_count += 1
                    continue
                
                X_rows.append(featurize(row))
            except Exception as e:
                print(f"Row {row_count}: Error processing: {e}")
                error_count += 1
                continue
        
        print(f"Local training: Processed {row_count} rows, {len(X_rows)} successful, {error_count} errors")
        
        if not X_rows:
            raise HTTPException(status_code=400, detail=f"No usable rows in CSV. Processed {row_count} rows, {error_count} errors. Training requires labeled data with 'isFraud' column.")
        
        X = np.vstack(X_rows)
        y = np.array(y_rows, dtype=float)
        mask = y != -1
        if mask.sum() == 0:
            raise HTTPException(status_code=400, detail="Dataset missing valid labels (isFraud)")
        
        # Prepare per-client local model path
        client_id = SESSIONS[token]
        local_model_path = os.path.join(MODELS_DIR, f"local_{client_id}.pt")

        # Initialize client's local model from GLOBAL on first run
        clf = LSTMClassifier(input_dim=global_model.input_dim)
        try:
            sd_local = load_torch(local_model_path)
            if sd_local is not None:
                clf.load_state_dict(sd_local)
                print(f"Loaded existing local model for client {client_id}")
            else:
                # initialize from current GLOBAL weights
                clf.load_state_dict(global_model.lstm.state_dict())
                print(f"Initialized local model from global weights for client {client_id}")
        except Exception as e:
            print(f"Error loading local model, using fresh model: {e}")
            # fallback to fresh model if load fails
            pass
        
        # Train the local model
        for epoch in range(max(1, int(epochs))):
            print(f"Training epoch {epoch + 1}/{epochs}")
            clf.train_one_epoch(X[mask], y[mask], batch_size=128)
        
        # Aggregate into GLOBAL and refresh IF
        print("Aggregating local model into global...")
        global_model.aggregate_lstm([clf.state_dict()])
        
        print("Updating global Isolation Forest...")
        global_model.update_iso(X)
        
        print("Saving global model...")
        global_model.save()

        # Persist/refresh client's local model copy
        try:
            save_torch(local_model_path, clf.state_dict())
            print(f"Saved local model for client {client_id}")
        except Exception as e:
            print(f"Warning: Could not save local model: {e}")
            pass

        return {"ok": True, "samples": int(X.shape[0]), "client_id": client_id, "epochs": int(epochs)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Local retrain failed: {e}")

# WebSocket stream endpoint
clients_ws: List[WebSocket] = []

@app.websocket("/stream")
async def stream_ws(websocket: WebSocket):
    await websocket.accept()
    clients_ws.append(websocket)
    try:
        # simple demo push loop: send a tx every 1/rate sec
        async for tx in stream_transactions(rate_hz=2.0):
            try:
                await websocket.send_text(json.dumps(tx))
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in clients_ws:
            clients_ws.remove(websocket)

# Run uvicorn programmatically for convenience if this file executed
def start_server(host="0.0.0.0", port=8000):
    uvicorn.run("server.api_http:app", host=host, port=port, reload=False, log_level="info")
