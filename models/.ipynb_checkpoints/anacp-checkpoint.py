import torch
import numpy as np
import torch.nn.functional as F
from collections import defaultdict

class IntermediateLayer:
    def __init__(self, D, input_dim, device, reg):
        self.D = D
        self.device = device
        self.W = torch.randn(D, input_dim, device=device)
        self.A = torch.zeros(D, D, device=device)
        self.Z_means = defaultdict(lambda: torch.zeros(D, device=self.device))
        self.counts = defaultdict(int)
        self.reg = reg
        self.theta = None

    def update_stats(self, X, Y):
        Z = F.gelu(X @ self.W.T)
        self.A += Z.T @ Z
        for c in torch.unique(Y):
            mask = (Y == c)
            c = c.item()
            z_mean = Z[mask].mean(dim=0)
            n = mask.sum().item()
            total = self.counts[c] + n
            self.Z_means[c] = (self.Z_means[c] * self.counts[c] + z_mean * n) / total
            self.counts[c] = total
        return Z

    def solve_theta(self, target_means):
        B = torch.zeros((self.D, target_means.shape[1]), device=self.device)
        for c, z_mean in self.Z_means.items():
            B += self.counts[c] * torch.outer(z_mean, target_means[c])
        self.theta = torch.linalg.solve(self.A + self.reg * torch.eye(self.D, device=self.device), B)

    def forward(self, X):
        Z = F.gelu(X @ self.W.T)
        return F.normalize(Z @ self.theta, p=2, dim=1)


class XLM:
    def __init__(self, D:int, reg:float, num_heads:int, seed:int, device:torch.device, samples_per_class, shared_cov:bool):
        self.device = device
        self.D = D
        self.num_heads = num_heads
        self.reg = reg
        self.seed = seed
        self.samples_per_class = samples_per_class
        self.shared_cov_flag = shared_cov
        if not shared_cov:
            self.class_covs = {}
        torch.manual_seed(self.seed)

        self.input_dim = None
        self.num_classes = 0
        self.class_counts = defaultdict(int)
        self.class_means = None
        self.class_map = {}
        self.inverse_map = {}

        self.theta = None
        self.W = None
        self.shared_cov = None

    def _normalize(self, X):
        return F.normalize(X, p=2, dim=1)

    def _update_shared_cov(self, X, Y_mapped):
        means = torch.stack([self.class_means[c.item()] for c in Y_mapped])
        X_centered = X - means
        cov_new = X_centered.T @ X_centered / X.shape[0]
        if self.shared_cov is None:
            self.shared_cov = cov_new + 1e-4 * torch.eye(self.input_dim, device=self.device)
        else:
            n = sum(self.class_counts.values())
            self.shared_cov = (self.shared_cov * n + cov_new * X.shape[0]) / (n + X.shape[0])

    def _spread_means(self, class_means, cov):
        eigvals, eigvecs = torch.linalg.eigh(cov)
        W = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + 1e-8)) @ eigvecs.T
        mean_center = class_means.mean(dim=0, keepdim=True)
        whitened = (class_means - mean_center) @ W.T
        U, S, Vh = torch.linalg.svd(whitened, full_matrices=False)
        spread_S = S + 1
        spread_centered = U @ torch.diag(spread_S) @ Vh
        adjusted_means = spread_centered @ W + mean_center
        return self._normalize(adjusted_means)

    def _update_class_map(self, Y):
        new_labels = sorted(set(Y.tolist()))
        for label in new_labels:
            if label not in self.class_map:
                mapped_id = self.num_classes
                self.class_map[label] = mapped_id
                self.inverse_map[mapped_id] = label
                self.num_classes += 1

    def _generate_replay_data(self):
        N = self.num_classes * self.samples_per_class
        replay_Y = torch.arange(self.num_classes, device=self.device).repeat_interleave(self.samples_per_class)
        class_means = torch.stack([self.class_means[c] for c in range(self.num_classes)])

        if self.shared_cov_flag:
            noise = torch.distributions.MultivariateNormal(torch.zeros(self.input_dim, device=self.device), self.shared_cov).sample((N,))
            replay_X = noise + class_means[replay_Y]
        else:
            chunks = []
            for c in range(self.num_classes):
                cov = self.class_covs[c]
                mean = class_means[c]
                noise = torch.distributions.MultivariateNormal(torch.zeros(self.input_dim, device=self.device), cov).sample((self.samples_per_class,))
                chunks.append(noise + mean)
            replay_X = torch.cat(chunks, dim=0)

        return replay_X, replay_Y

    def _update_class_cov(self, X, Y_mapped):
        for c in torch.unique(Y_mapped):
            mask = (Y_mapped == c)
            c = c.item()
            centered = X[mask] - self.class_means[c]
            cov = centered.T @ centered / centered.shape[0]
            if c in self.class_covs:
                n = self.class_counts[c]
                n_prev = n - centered.shape[0]
                self.class_covs[c] = (self.class_covs[c] * n_prev + cov * centered.shape[0]) / n
            else:
                self.class_covs[c] = cov + 1e-4 * torch.eye(self.input_dim, device=self.device)

    def update(self, X: np.ndarray, Y: np.ndarray):
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = self._normalize(X)
        Y = torch.tensor(Y, dtype=torch.long, device=self.device)

        if self.input_dim is None:
            self.input_dim = X.shape[1]
            self.layers = [IntermediateLayer(self.D, self.input_dim, self.device, self.reg) for _ in range(self.num_heads)]
            self.class_means = defaultdict(lambda: torch.zeros(self.input_dim, device=self.device))

        self._update_class_map(Y)
        Y_mapped = torch.tensor([self.class_map[y.item()] for y in Y], device=self.device)

        for c in torch.unique(Y_mapped):
            c_int = c.item()
            x_mean = X[Y_mapped == c].mean(dim=0)
            n = (Y_mapped == c).sum().item()
            total = self.class_counts[c_int] + n
            self.class_means[c_int] = (self.class_means[c_int] * self.class_counts[c_int] + x_mean * n) / total
            self.class_counts[c_int] = total

        self._update_shared_cov(X, Y_mapped)
        if not self.shared_cov_flag:
            self._update_class_cov(X, Y_mapped)
        stacked_means = torch.stack([self.class_means[c] for c in range(self.num_classes)])
        target_means = self._spread_means(stacked_means, self.shared_cov)

        for layer in self.layers:
            layer.update_stats(X, Y_mapped)
            layer.solve_theta(target_means)

        replay_X, replay_Y = self._generate_replay_data()
        outputs = [layer.forward(replay_X) for layer in self.layers]

        outs = torch.stack(outputs).mean(dim=0)
        self.W = torch.randn(self.D, self.input_dim, device=self.device)
        H = F.gelu(outs @ self.W.T)

        Y_onehot = torch.zeros(replay_Y.size(0), self.num_classes, device=self.device)
        Y_onehot.scatter_(1, replay_Y.unsqueeze(1), 1.0)

        A = H.T @ H + self.reg * torch.eye(self.D, device=self.device)
        B = H.T @ Y_onehot
        self.theta = torch.linalg.solve(A, B)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = torch.tensor(X, dtype=torch.float32, device=self.device)
        X = self._normalize(X)

        outputs = [layer.forward(X) for layer in self.layers]
        outs = torch.stack(outputs).mean(dim=0)

        H = F.gelu(outs @ self.W.T)
        logits = H @ self.theta
        preds = torch.argmax(logits, dim=1)
        return np.array([self.inverse_map[p.item()] for p in preds])
