# run.py
from server.main_http import start_server

if __name__ == "__main__":
    # Start FastAPI (uvicorn) server
    start_server(host="0.0.0.0", port=8000)
