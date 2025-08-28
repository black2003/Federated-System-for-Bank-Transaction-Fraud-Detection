import os
import hashlib
import secrets
from typing import Dict

from .utils.serialization import STORAGE_DIR, load_json, save_json

CREDS_PATH = os.path.join(STORAGE_DIR, "credentials.json")

class AuthDB:
    def __init__(self):
        os.makedirs(STORAGE_DIR, exist_ok=True)
        self.db = load_json(CREDS_PATH, default={"admin": {}, "clients": {}})

    def _hash(self, password: str, salt: str) -> str:
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000)
        return dk.hex()

    def ensure_admin(self) -> None:
        if "hash" not in self.db.get("admin", {}):
            print("No admin password set. Please create one now.")
            while True:
                pwd = input("Set admin password: ").strip()
                if len(pwd) < 6:
                    print("Password must be at least 6 chars.")
                    continue
                salt = secrets.token_hex(16)
                self.db["admin"] = {"salt": salt, "hash": self._hash(pwd, salt)}
                self._persist()
                print("Admin password set.\n")
                break

    def has_admin(self) -> bool:
        adm = self.db.get("admin", {})
        return "hash" in adm and "salt" in adm

    def set_admin(self, password: str) -> bool:
        """Bootstrap admin password if not already set. Returns True if set, False if exists."""
        if self.has_admin():
            return False
        if not password or len(password) < 6:
            return False
        salt = secrets.token_hex(16)
        self.db["admin"] = {"salt": salt, "hash": self._hash(password, salt)}
        self._persist()
        return True

    def verify_admin(self, password: str) -> bool:
        adm = self.db.get("admin", {})
        if "hash" not in adm:
            return False
        return self._hash(password, adm["salt"]) == adm["hash"]

    def register_client(self, client_id: str, name: str, password: str) -> bool:
        clients = self.db.setdefault("clients", {})
        if client_id in clients:
            return False
        salt = secrets.token_hex(16)
        clients[client_id] = {"name": name, "salt": salt, "hash": self._hash(password, salt)}
        self._persist()
        return True

    def delete_client(self, client_id: str) -> bool:
        clients = self.db.setdefault("clients", {})
        if client_id not in clients:
            return False
        del clients[client_id]
        self._persist()
        return True

    def verify_client(self, client_id: str, password: str) -> bool:
        c = self.db.get("clients", {}).get(client_id)
        if not c:
            return False
        return self._hash(password, c["salt"]) == c["hash"]

    def list_clients(self) -> Dict[str, Dict]:
        return self.db.get("clients", {})

    def _persist(self):
        save_json(CREDS_PATH, self.db)
