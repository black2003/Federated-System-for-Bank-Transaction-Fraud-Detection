# Federated Fraud Detection System (FEDv3)

A real-time federated learning system for fraud detection with enhanced client dashboard, admin console, and ZKP integration.

## Features

- **Real-time Fraud Detection**: Live transaction streaming with instant fraud classification
- **Federated Learning**: Client-side training with global model aggregation
- **Enhanced UI**: Modern client dashboard with dark mode, collapsible sections, and export features
- **Admin Console**: Client management, global model training, and system monitoring
- **ZKP Integration**: Zero-knowledge proof export for transaction verification
- **Cross-platform**: Connect clients from different machines/IPs

## Quick Start

### 1. Install Dependencies

```bash
# From project root
pip install -r requirements.txt

# Or using conda
conda install --file requirements.txt
```

### 2. Start the Server

```bash
python run.py
```

Server runs on `http://0.0.0.0:8000`

### 3. Access the System

- **Admin Console**: `http://localhost:8000/admin`
- **Client Dashboard**: `http://localhost:8000/client`

## Detailed Setup

### Prerequisites

- Python 3.8+ (3.10+ recommended)
- pip or conda package manager
- Modern web browser

### Installation Steps

1. **Clone/Download the project**
2. **Create virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Start the server**
   ```bash
   python run.py
   ```

### First-Time Setup

1. **Open Admin Console**: `http://localhost:8000/admin`
2. **Set Admin Password**: Click "Bootstrap Admin Password"
3. **Add a Client**: Use the "Add Client" form with:
   - Client ID (e.g., "bank1")
   - Name (e.g., "First Bank")
   - Password (e.g., "password123")
   - Admin password (the one you set in step 2)

## Usage

### Admin Console

- **Client Management**: Add/remove clients, view registry
- **Global Training**: Upload labeled CSV datasets to train the global model
- **System Status**: Monitor server connectivity and ZKP status

### Client Dashboard

1. **Connect to Server**
   - Set Server URL (e.g., `http://192.168.1.100:8000` for remote)
   - Click "Apply"
   - Login with your client credentials

2. **Live Stream**
   - Click "Connect Stream" to start real-time fraud detection
   - Adjust threshold slider for fraud sensitivity
   - Use Pause/Resume, Clear, and Export controls
   - Enable "Export to ZKP" for ZKP integration

3. **Dataset Evaluation**
   - Upload unlabeled CSV for fraud detection
   - View fraud counts and get alerts
   - Export predictions to CSV

4. **Local Training**
   - Upload labeled CSV (with `isFraud` column)
   - Set training epochs
   - Train locally and push updates to global model

### ZKP Integration (Optional)

1. **Start ZKP API** (if available):
   ```bash
   cd ZKP/MitsubishiProd
   uvicorn api.main:app --host 0.0.0.0 --port 7000
   ```

2. **Enable ZKP Export**: Check "Export to ZKP" in client dashboard
3. **Verify**: Check `/zkp/status` endpoint or ZKP API

## Configuration

### Server Settings

- **Host**: `0.0.0.0` (accessible from any IP)
- **Port**: `8000`
- **CORS**: Enabled for cross-origin requests

### Client Settings

- **Server URL**: Configurable for remote connections
- **Threshold**: Adjustable fraud detection sensitivity
- **Auto-reconnect**: WebSocket auto-reconnection
- **Dark Mode**: Persistent UI theme preference

## File Structure

```
FEDv3/
├── server/                 # Main server implementation
│   ├── api_http.py        # FastAPI endpoints and WebSocket
│   ├── federation.py      # Global model management
│   ├── models/            # ML models (LSTM, Isolation Forest)
│   ├── templates/         # HTML templates
│   └── static/            # JavaScript and CSS
├── client/                # Client implementations
├── storage/               # Model storage and client registry
├── ZKP/                   # Zero-knowledge proof integration
├── run.py                 # Server entry point
└── requirements.txt       # Python dependencies
```

## API Endpoints

### Core Endpoints
- `POST /login` - Client authentication
- `POST /predict` - Batch fraud prediction
- `POST /client/evaluate_dataset` - CSV evaluation
- `POST /client/local_retrain_dataset` - Local model training
- `POST /push_update` - Model update aggregation

### Admin Endpoints
- `GET /admin/clients` - List registered clients
- `POST /admin/register_client` - Add new client
- `POST /admin/remove_client` - Remove client
- `POST /admin/train_global_dataset` - Global model training

### ZKP Endpoints
- `POST /zkp/export_tx` - Export transaction to ZKP
- `GET /zkp/status` - ZKP system status

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

2. **Connection Issues**
   - Verify server is running: `python run.py`
   - Check firewall settings
   - Use correct Server URL in client dashboard

3. **ZKP Integration**
   - Ensure ZKP directory exists and is accessible
   - Check ZKP API status at `/zkp/status`

4. **Model Training**
   - Verify CSV format (required columns: amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type)
   - For local training: include `isFraud` column

### Logs and Debugging

- Server logs appear in console when running `python run.py`
- Client errors shown in browser console (F12)
- Check `/zkp/status` for ZKP system health

## Development

### Adding New Features

1. **Server**: Modify `server/api_http.py`
2. **Client UI**: Edit `server/templates/client.html`
3. **Admin UI**: Edit `server/templates/dashboard.html`
4. **Models**: Add to `server/models/`

### Testing

```bash
# Install dev dependencies
pip install -r requirements.txt

# Run tests (if available)
pytest

# Manual testing
python run.py
# Open http://localhost:8000/admin and http://localhost:8000/client
```

## License

This project is for educational and research purposes.

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Check server logs for error messages
4. Ensure proper file permissions and paths
