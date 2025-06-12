import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
import gpytorch
from gpytorch.mlls import VariationalELBO
import numpy as np
from sklearn.model_selection import train_test_split
from ..utils.pair_dataset import PairDataset
from torch import optim

class RBFModule(nn.Module):
    def __init__(self, feature_extractor, d=200):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.d = d
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        f1 = self.feature_extractor(x1)
        f2 = self.feature_extractor(x2)
        covar = self.covar_module(f1, f2).evaluate()
        # print(f"Covariance shape: {covar.shape}")
        return covar

class DeepKernelLearning():
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        lr: float = 0.1,
        epochs: int = 100,
        batch_size: int = 64,
        use_feature_extractor: bool = True,
        device=None,
        patience: int = 10,
    ):
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_fe = use_feature_extractor
        self.X_train = None
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.patience = patience

        # 1) optional MLP feature extractor
        if self.use_fe:
            self.feature_extractor = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            feat_dim = hidden_dim
        else:
            # identity mapping
            self.feature_extractor = lambda x: x
            feat_dim = input_dim

    def fit(self, Xs_train, Y_train, Xs_test=None, Y_test=None):
        self.X_train = Xs_train.copy()
        input_dim = Xs_train.shape[1]

        self.model = RBFModule(self.feature_extractor, input_dim).to(self.device)
        
        X_train = torch.tensor(Xs_train, dtype=torch.float32).to(self.device)
        Y_train = torch.tensor(Y_train, dtype=torch.float32).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        print("Y_train ", Y_train)
        def kernel_alignment_loss(K_pred, K_true):
            K_pred = (K_pred - K_pred.mean()) / (K_pred.std() + 1e-8)
            K_true = (K_true - K_true.mean()) / (K_true.std() + 1e-8)
            alignment = (K_pred * K_true).sum() / (K_pred.norm() * K_true.norm() + 1e-8)
            return 1 - alignment
    
        Y_mat = Y_train.unsqueeze(0) == Y_train.unsqueeze(1)
        K_true = Y_mat.float().to(self.device)
        print(f"True kernel shape: {K_true.shape}")
        print(f"True kernel: {K_true}")
        # Kiểm tra các giá trị duy nhất trong K_true
        print(f"Unique values in K_true: {torch.unique(K_true).cpu().numpy()}")
        for i in range(self.epochs):
            self.model.train()
            optimizer.zero_grad()

            K_pred = self.model(X_train, X_train)
            if i % 190 == 0:
                print(f"Predicted kernel shape: {K_pred.shape}")
                print(f"Predicted kernel: {K_pred}")
            loss = kernel_alignment_loss(K_pred, K_true)
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"Iter {i}/{self.epochs}, Loss: {loss.item():.4f}")

    def transform(self, Xs_test):
        # print("► In DeepKernelLearning.transform, self.X_train shape =", np.shape(self.X_train))
        self.model.eval()
        with torch.no_grad():
            Xs_test = torch.tensor(Xs_test, dtype=torch.float32).to(self.device)
            X_train = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
            K_test = self.model(Xs_test, X_train) #not working in train vs test
            return K_test.cpu().numpy()