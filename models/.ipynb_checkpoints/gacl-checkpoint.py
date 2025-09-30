import torch
import numpy as np
import torch.nn.functional as F

class RanPac:
    def __init__(self, D: int, reg: float, seed: int, device: torch.device):
        self.num_features = None
        self.D = D
        self.reg = reg
        self.seed = seed
        self.device = device
        
        self.G = None
        self.C = None
        self.Theta = None
        self.is_updated = False

        self.num_classes = 0
        self.label_to_index = {}
        self.index_to_label = {}

    def _init_projection(self, input_dim):
        torch.manual_seed(self.seed)
        self.W = torch.randn(self.D, input_dim, device=self.device)  # random projection    

    def _update_theta(self):
        reg_matrix = self.reg * torch.eye(self.D, device=self.device)
        self.Theta = torch.linalg.solve(self.G + reg_matrix, self.C)
        self.is_updated = True

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.tensor(Y, dtype=torch.long, device=self.device)

        if self.num_features is None:
            self.num_features = X.shape[1]
            self._init_projection(self.num_features)

        # Map labels to indices
        new_labels = [y.item() for y in Y.unique() if y.item() not in self.label_to_index]
        for label in new_labels:
            self.label_to_index[label] = self.num_classes
            self.index_to_label[self.num_classes] = label
            self.num_classes += 1

        label_indices = torch.tensor([self.label_to_index[y.item()] for y in Y], device=self.device)

        H = torch.relu(X @ self.W.T)  # shape: (N, D)
        N = H.size(0)

        if self.C is None:
            self.C = torch.zeros(self.D, self.num_classes, device=self.device)
        elif self.C.size(1) < self.num_classes:
            new_C = torch.zeros(self.D, self.num_classes, device=self.device)
            new_C[:, :self.C.size(1)] = self.C
            self.C = new_C

        if self.G is None:
            self.G = H.T @ H
        else:
            self.G += H.T @ H

        Y_onehot = torch.zeros(N, self.num_classes, device=self.device)
        Y_onehot.scatter_(1, label_indices.unsqueeze(1), 1.0)
        self.C += H.T @ Y_onehot

        self.is_updated = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        H = torch.relu(X @ self.W.T)

        if not self.is_updated:
            self._update_theta()

        logits = H @ self.Theta  # shape: (N, C)
        preds = torch.argmax(logits, dim=1)
        return np.array([self.index_to_label[i.item()] for i in preds])
