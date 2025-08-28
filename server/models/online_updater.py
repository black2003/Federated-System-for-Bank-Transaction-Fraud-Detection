from collections import deque
import numpy as np

class OnlineBuffer:
    def __init__(self, maxlen: int = 5000):
        self.X = deque(maxlen=maxlen)
        self.y = deque(maxlen=maxlen)

    def add(self, X_row, y_row=None):
        self.X.append(X_row)
        self.y.append(y_row if y_row is not None else -1)

    def batch(self):
        if not self.X:
            return None, None
        X = np.array(self.X)
        y = np.array(self.y)
        return X, y

    def labeled_batch(self):
        if not self.X:
            return None, None
        X = []
        y = []
        for xi, yi in zip(self.X, self.y):
            if yi is not None and yi != -1:
                X.append(xi)
                y.append(yi)
        if not X:
            return None, None
        return (np.array(X), np.array(y))
