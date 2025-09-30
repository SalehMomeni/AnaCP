import torch
import numpy as np
from collections import defaultdict

class SLDA:
    def __init__(self, reg: float, device: torch.device):
        self.reg = reg
        self.device = device
        self.num_features = None

        self.class_means = {}
        self.class_counts = defaultdict(int)
        self.cov = None
        self.total_count = 0

        self.cov_inv = None
        self.is_updated = False

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.int64, device=self.device)

        if self.num_features is None:
            self.num_features = X.shape[1]
            self.cov = torch.zeros((self.num_features, self.num_features), device=self.device)

        for cls in Y.unique():
            cls_mask = (Y == cls)
            X_cls = X[cls_mask]
            n_new = X_cls.size(0)
            cls_mean_new = X_cls.mean(dim=0)

            n_old = self.class_counts[cls.item()]
            if n_old == 0:
                self.class_means[cls.item()] = cls_mean_new
            else:
                cls_mean_old = self.class_means[cls.item()]
                updated_mean = (n_old * cls_mean_old + n_new * cls_mean_new) / (n_old + n_new)
                self.class_means[cls.item()] = updated_mean

            self.class_counts[cls.item()] += n_new

            centered = X_cls - self.class_means[cls.item()]
            self.cov += centered.T @ centered
            self.total_count += n_new

        self.is_updated = False

    def _update_cov_inv(self):
        cov = self.cov / (self.total_count - len(self.class_means))
        cov += self.reg * torch.eye(self.num_features, device=self.device)
        self.cov_inv = torch.inverse(cov)
        self.is_updated = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32, device=self.device)

        if not self.is_updated:
            self._update_cov_inv()

        class_labels = list(self.class_means.keys())
        means = torch.stack([self.class_means[cls] for cls in class_labels])  # (C, d)

        # LDA formulation: x^T Σ^{-1} μ - 0.5 μ^T Σ^{-1} μ
        x_proj = X @ self.cov_inv @ means.T  # (N, C)
        mean_proj = 0.5 * torch.sum((means @ self.cov_inv) * means, dim=1)  # (C,)
        scores = x_proj - mean_proj  # (N, C)

        preds = scores.argmax(dim=1)
        return np.array([class_labels[i] for i in preds.cpu().numpy()])
