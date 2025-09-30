import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

class NCM:
    def __init__(self, device: torch.device):
        self.device = device
        self.input_dim = None
        self.class_sums = None
        self.class_counts = defaultdict(int)

    @staticmethod
    def normalize(X):
        return F.normalize(X, dim=1)

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.long, device=self.device)

        if self.input_dim is None:
            self.input_dim = X.shape[1]
            self.class_sums = defaultdict(lambda: torch.zeros(self.input_dim, device=self.device))

        for cls in torch.unique(Y):
            cls_mask = (Y == cls)
            X_cls = X[cls_mask]
            self.class_sums[int(cls)] += X_cls.sum(dim=0)
            self.class_counts[int(cls)] += X_cls.size(0)

    def predict(self, X: np.ndarray, Y: np.ndarray = None):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = self.normalize(X)
        
        class_ids = list(self.class_sums.keys())
        means = torch.stack([self.class_sums[c] / self.class_counts[c] for c in class_ids])  # (C, d)
        means = self.normalize(means)
        
        sims = torch.matmul(X, means.t())  # (N, C)
        preds = torch.argmax(sims, dim=1)
        return np.array([class_ids[p.item()] for p in preds])
