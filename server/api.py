import asyncio
import os
from typing import Dict, List, Tuple
import numpy as np

from .auth import AuthDB
from .federation import GlobalModel
from .data.stream import stream_transactions, generate_batch
from .utils.logging_config import setup_logging

logger = setup_logging()

FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "delta_org", "delta_dest"
]

TYPE_MAP = {"PAYMENT":0, "TRANSFER":1, "CASH_OUT":2, "DEBIT":3, "CASH_IN":4}

def featurize(tx: Dict) -> np.ndarray:
    delta_org = tx["oldbalanceOrg"] - tx["newbalanceOrig"]
    delta_dest = tx["newbalanceDest"] - tx["oldbalanceDest"]
    base = [
        tx["amount"], tx["oldbalanceOrg"], tx["newbalanceOrig"],
        tx["oldbalanceDest"], tx["newbalanceDest"], delta_org, delta_dest
    ]
    one_hot = [0]*5
    one_hot[TYPE_MAP.get(tx["type"], 0)] = 1
    return np.array(base + one_hot, dtype=float)

INPUT_DIM = 7 + 5  # numeric + type one-hot

class Registry:
    def __init__(self, path: str = os.path.join("storage", "registry.json")):
        self.path = path
        self.state = self._load()

    def _load(self):
        if not os.path.exists(self.path):
            return {"clients": {}}
        import json
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self):
        import json
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def add(self, client_id: str, name: str):
        self.state.setdefault("clients", {})[client_id] = {"name": name}
        self._save()

    def remove(self, client_id: str):
        self.state.setdefault("clients", {}).pop(client_id, None)
        self._save()

    def list(self):
        return self.state.get("clients", {})

class ServerApp:
    def __init__(self):
        self.auth = AuthDB()
        self.registry = Registry()
        self.global_model = GlobalModel(input_dim=INPUT_DIM)
        self.global_model.load()
        self.stream_task = None
        self.stream_running = False
        self.rate_hz = 5.0

    async def run(self):
        self.auth.ensure_admin()
        print("\n=== Realtime Fraud Detection — Server ===")
        while True:
            print("\nMain Menu:\n 1) Start/Stop stream\n 2) Client login\n 3) Admin menu\n 4) Exit")
            choice = input("> ").strip()
            if choice == "1":
                await self.toggle_stream()
            elif choice == "2":
                await self.client_login()
            elif choice == "3":
                await self.admin_menu()
            elif choice == "4":
                print("Bye!")
                break
            else:
                print("Invalid option.")

    async def toggle_stream(self):
        if self.stream_running:
            print("Stopping stream...")
            if self.stream_task:
                self.stream_task.cancel()
                try:
                    await self.stream_task
                except asyncio.CancelledError:
                    pass
            self.stream_running = False
        else:
            print(f"Starting stream at {self.rate_hz} tx/s ... (clients pull batches on demand)")
            self.stream_task = asyncio.create_task(self._run_stream())
            self.stream_running = True

    async def _run_stream(self):
        async for tx in stream_transactions(rate_hz=self.rate_hz):
            logger.debug(f"Stream step={tx['step']} type={tx['type']} amount={tx['amount']:.2f}")

    async def client_login(self):
        cid = input("Client ID: ").strip()
        pwd = input("Password: ").strip()
        if not self.auth.verify_client(cid, pwd):
            print("\nLogin failed.\n")
            return
        name = self.auth.db["clients"][cid]["name"]
        # Lazy import to avoid hard dependency during FastAPI startup
        from client.menu import client_menu
        await client_menu(cid, name, self)

    async def admin_menu(self):
        pwd = input("Admin password: ").strip()
        if not self.auth.verify_admin(pwd):
            print("Unauthorized.\n")
            return
        while True:
            print("\nAdmin Menu:\n 1) Register client\n 2) Remove client\n 3) List clients\n 4) Save global model\n 5) Back")
            ch = input("> ").strip()
            if ch == "1":
                cid = input("New client ID: ").strip()
                name = input("Client name: ").strip()
                pw = input("Set client password: ").strip()
                ok = self.auth.register_client(cid, name, pw)
                if ok:
                    self.registry.add(cid, name)
                    print("Client registered.")
                else:
                    print("Client ID already exists.")
            elif ch == "2":
                cid = input("Client ID to remove: ").strip()
                ok = self.auth.delete_client(cid)
                if ok:
                    self.registry.remove(cid)
                    print("Client removed.")
                else:
                    print("Client not found.")
            elif ch == "3":
                clients = self.registry.list()
                if not clients:
                    print("(no clients)")
                else:
                    for k, v in clients.items():
                        print(f"- {k} :: {v['name']}")
            elif ch == "4":
                self.global_model.save()
                print("Global model saved.")
            elif ch == "5":
                break
            else:
                print("Invalid option.")

    # ---- Utilities used by clients ----

    def fetch_synthetic_batch(self, n: int, start_step: int = 1):
        return generate_batch(n, start_step)

    def featurize_batch(self, txs: List[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        X = np.vstack([featurize(t) for t in txs])
        y = np.array([t["isFraud"] for t in txs], dtype=float)
        return X, y
