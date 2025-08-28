# server/models/lstm_model.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class TinyLSTM(nn.Module):
    def __init__(self, input_dim:int, hidden:int=32, num_layers:int=1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, T, D)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        logits = self.head(last)
        return self.sigmoid(logits).squeeze(-1)

class LSTMClassifier:
    def __init__(self, input_dim:int, lr:float=1e-3):
        self.input_dim = input_dim
        self.net = TinyLSTM(input_dim)
        self.criterion = nn.BCELoss()
        self.opt = optim.Adam(self.net.parameters(), lr=lr)
        self.device = torch.device("cpu")
        self.net.to(self.device)

    def predict_proba(self, X: np.ndarray):
        # X: (N,D) or (N,1,D)
        self.net.eval()
        if X is None or len(X)==0:
            return np.zeros((0,), dtype=float)
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            if X_t.ndim == 2:
                X_t = X_t.unsqueeze(1)  # (N,1,D)
            out = self.net(X_t).cpu().numpy()
        return out

    def train_one_epoch(self, X: np.ndarray, y: np.ndarray, batch_size:int=64):
        if X is None or len(X)==0:
            return
        self.net.train()
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        if X_t.ndim == 2:
            X_t = X_t.unsqueeze(1)
        y_t = torch.tensor(y, dtype=torch.float32, device=self.device)
        N = X_t.size(0)
        idx = torch.randperm(N)
        for i in range(0, N, batch_size):
            sel = idx[i:i+batch_size]
            xb = X_t[sel]
            yb = y_t[sel]
            self.opt.zero_grad()
            pred = self.net(xb)
            loss = self.criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
            self.opt.step()

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)
