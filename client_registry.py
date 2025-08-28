# client_registry.py
import json
import os
from threading import Lock

REGISTRY_FILE = "client_registry.json"


class ClientRegistry:
    def __init__(self):
        self._lock = Lock()
        self.clients = {}
        self.load()

    def load(self):
        """Load registry from disk"""
        if os.path.exists(REGISTRY_FILE):
            try:
                with open(REGISTRY_FILE, "r") as f:
                    self.clients = json.load(f)
            except Exception:
                self.clients = {}
        else:
            self.clients = {}

    def save(self):
        """Persist registry to disk"""
        with self._lock:
            with open(REGISTRY_FILE, "w") as f:
                json.dump(self.clients, f, indent=4)

    def add_client(self, client_id, info=None):
        """Register a new client"""
        with self._lock:
            if client_id in self.clients:
                return False  # already exists
            self.clients[client_id] = info or {}
            self.save()
            return True

    def remove_client(self, client_id):
        """Remove a client"""
        with self._lock:
            if client_id in self.clients:
                del self.clients[client_id]
                self.save()
                return True
            return False

    def update_client(self, client_id, info):
        """Update client metadata"""
        with self._lock:
            if client_id in self.clients:
                self.clients[client_id].update(info)
                self.save()
                return True
            return False

    def list_clients(self):
        """Return all registered clients"""
        return self.clients

    def get_client(self, client_id):
        return self.clients.get(client_id, None)
