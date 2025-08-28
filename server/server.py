from flask import Flask, request, jsonify, render_template
from server.models.fusion import FusionModel
from client_registry import ClientRegistry
import config
from flask_socketio import SocketIO

app = Flask(__name__)
registry = ClientRegistry()
global_model = FusionModel()
app.config['SECRET_KEY'] = 'supersecretkey'
socketio = SocketIO(app, cors_allowed_origins="*")


# --- AUTH HELPERS ---
def check_admin(user, password):
    return user == config.ADMIN_USERNAME and password == config.ADMIN_PASSWORD


@app.route("/")
def dashboard():
    return render_template("dashboard.html", clients=registry.list_clients())


# --- ADMIN: ADD CLIENT ---
@app.route("/admin/add_client", methods=["POST"])
def add_client():
    data = request.json
    if not check_admin(data.get("username"), data.get("password")):
        return jsonify({"error": "Unauthorized"}), 403

    client_id = data["client_id"]
    name = data["name"]
    password = data["password"]

    # Ensure unique client id
    if registry.get_client(client_id):
        return jsonify({"error": "Client already exists"}), 400

    # Add client with its own FusionModel
    registry.add_client(client_id, name, password)
    registry.update_client(client_id, {"model": FusionModel()})

    return jsonify({"status": "Client added", "clients": registry.list_clients()})


# --- ADMIN: REMOVE CLIENT ---
@app.route("/admin/remove_client/<client_id>", methods=["POST"])
def remove_client(client_id):
    data = request.json
    if not check_admin(data.get("username"), data.get("password")):
        return jsonify({"error": "Unauthorized"}), 403

    registry.remove_client(client_id)
    return jsonify({"status": "removed", "clients": registry.list_clients()})


# --- CLIENT: LOCAL TRAINING ---
@app.route("/client/train_local/<client_id>", methods=["POST"])
def client_train_local(client_id):
    client = registry.get_client(client_id)
    if not client:
        return jsonify({"error": "Client not found"}), 404

    model = client.get("model")
    if not model:
        return jsonify({"error": "No local model found"}), 400

    # Train local model (stub - replace with dataset loading later)
    history = model.train_local()
    return jsonify(
        {"status": "local training done", "client_id": client_id, "history": history}
    )


# --- ADMIN: TRAIN GLOBAL ---
@app.route("/admin/train_global", methods=["POST"])
def train_global():
    data = request.json
    if not check_admin(data.get("username"), data.get("password")):
        return jsonify({"error": "Unauthorized"}), 403

    global global_model
    global_model.aggregate_from_clients(registry.clients)
    return jsonify({"status": "global model updated"})
