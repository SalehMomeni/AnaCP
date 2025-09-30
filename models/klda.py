import torch
import numpy as np
from collections import defaultdict

class KLDA:
    def __init__(self, D: int, gamma: float, reg: float, seed: int, device: torch.device):
        self.D = D
        self.gamma = gamma
        self.reg = reg
        self.seed = seed
        self.device = device

        self.num_features = None
        self.W = None  # RFF weight
        self.b = None  # RFF bias

        self.class_means = {}
        self.class_counts = defaultdict(int)
        self.sigma = None
        self.total_count = 0

        self.sigma_inv = None
        self.is_updated = False

    def _compute_rff(self, X):
        if self.W is None:
            input_dim = X.shape[1]
            torch.manual_seed(self.seed)
            self.W = torch.randn(input_dim, self.D, device=self.device) * np.sqrt(2 * self.gamma)
            self.b = 2 * np.pi * torch.rand(self.D, device=self.device)
        projection = X @ self.W + self.b
        return torch.sqrt(torch.tensor(2.0 / self.D, device=self.device)) * torch.cos(projection)

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.int64, device=self.device)

        Z = self._compute_rff(X)

        if self.num_features is None:
            self.num_features = Z.shape[1]
            self.sigma = torch.zeros((self.num_features, self.num_features), device=self.device)

        for cls in Y.unique():
            cls_mask = (Y == cls)
            Z_cls = Z[cls_mask]
            n_new = Z_cls.size(0)
            cls_mean_new = Z_cls.mean(dim=0)

            n_old = self.class_counts[cls.item()]
            if n_old == 0:
                self.class_means[cls.item()] = cls_mean_new
            else:
                cls_mean_old = self.class_means[cls.item()]
                updated_mean = (n_old * cls_mean_old + n_new * cls_mean_new) / (n_old + n_new)
                self.class_means[cls.item()] = updated_mean

            self.class_counts[cls.item()] += n_new

            centered = Z_cls - self.class_means[cls.item()]
            self.sigma += centered.T @ centered
            self.total_count += n_new

        self.is_updated = False

    def _update_sigma_inv(self):
        sigma = self.sigma + self.reg * torch.eye(self.D, device=self.device)
        self.sigma_inv = torch.inverse(sigma)
        self.is_updated = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Z = self._compute_rff(X)

        if not self.is_updated:
            self._update_sigma_inv()

        class_labels = list(self.class_means.keys())
        means = torch.stack([self.class_means[cls] for cls in class_labels])  # (C, D)

        # Mahalanobis distance: (x - mu)^T Sigma^{-1} (x - mu)
        Z_exp = Z.unsqueeze(1)  # (N, 1, D)
        means_exp = means.unsqueeze(0)  # (1, C, D)
        delta = Z_exp - means_exp  # (N, C, D)
        dists = torch.einsum("ncd,df,ncf->nc", delta, self.sigma_inv, delta)  # (N, C)
        preds = dists.argmin(dim=1)

        return np.array([class_labels[i] for i in preds.cpu().numpy()])
        