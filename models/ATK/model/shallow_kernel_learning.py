import numpy as np
from ..loss import my_loss
import torch
from cvxopt import matrix
import torch.nn as nn
import torch.optim as optim
import random
from ..utils.pair_dataset import PairDataset
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Cố định behavior của cuDNN (nếu dùng GPU) để đảm bảo tính tái lập
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Gọi hàm này ngay từ đầu chương trình
set_seed(4)     
# Định nghĩa mô hình RBFNet với 2 layers:
# - Layer 1: Tính RBF kernels với các tham số gamma học được (lưu dưới dạng log_gamma để đảm bảo > 0)
# - Layer 2: Kết hợp các đầu ra của RBF kernels với trọng số và bias, sau đó áp dụng sigmoid
class RBFNet(nn.Module):
    def __init__(self,feature_extractor, num_kernels=10):
        super(RBFNet, self).__init__()
        self.num_kernels = num_kernels
        # Lưu log_gamma để đảm bảo gamma = exp(log_gamma) luôn dương
        self.log_gamma = nn.Parameter(torch.randn(num_kernels))

        # Trọng số RBF dương bằng log_weights
        self.log_weights_rbf = nn.Parameter(torch.empty(num_kernels))

        nn.init.xavier_uniform_(self.log_weights_rbf.unsqueeze(0))
        self.bias = nn.Parameter(torch.zeros(1))
        self.feature_extractor = feature_extractor

        # self.linear_rbf = nn.Linear(num_kernels, 1)
        
    def forward(self, x, y):
        """
        x, y: tensor có shape [batch_size, input_dim]
        """
        # print(f"Input shape: {x.shape}, {y.shape}")
        x = self.feature_extractor(x)
        y = self.feature_extractor(y)
        # print(f"Feature shape: {x.shape}, {y.shape}")
        # Tính khoảng cách bình phương giữa x và y theo từng sample
        diff = torch.sum((x - y)**2, dim=1)  # [batch_size]
        diff_expanded = diff.unsqueeze(1)     # [batch_size, 1]
        # Lấy gamma từ log_gamma
        gamma = torch.exp(self.log_gamma)       # [num_kernels]
        weights_rbf = torch.exp(self.log_weights_rbf)  # Đảm bảo dương

        # Tính các giá trị RBF: exp(-gamma * ||x-y||^2)
        phi_rbf = torch.exp(- gamma * diff_expanded)  # [batch_size, num_kernels]
        
        # Kết hợp theo trọng số và bias
        out_rbf = torch.matmul(phi_rbf, weights_rbf) + self.bias  # [batch_size]
 
        # Áp dụng sigmoid để đưa về khoảng [0,1]
        prob = torch.sigmoid(out_rbf)


        # out_rbf = self.linear_rbf(phi_rbf)            # [batch_size, 1]
        # out_rbf = torch.sum(out_rbf, dim=1)  # [batch_size]
        # prob = torch.sigmoid(out_rbf.squeeze(-1))     # [batch_size]
        return prob

class ShallowKernelLearning():
    def __init__(self, hidden_dim: int=64 , 
                 num_kernels=10, lr=0.01, 
                 epochs=1500, pretrain_echops=300,
                 batch_size=64,lambda_reg=1e-3,
                 patience=10, use_fe:bool=False,
                 device=None):
        """
        Khởi tạo với các hyperparameter:
         - num_kernels: số RBF kernels (mặc định 10)
         - lr: learning rate cho optimizer
         - epochs: số epoch huấn luyện
         - batch_size: kích thước batch cho việc sampling cặp dữ liệu
         - device: thiết bị tính toán (GPU nếu có)
        """
        self.num_kernels = num_kernels
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_train = None  # Lưu lại tập train để dùng trong transform
        self.learned_gamma = None
        self.learned_weights = None
        self.learned_bias = None
        self.lambda_reg = lambda_reg
        self.patience = patience
        self.use_fe = use_fe
        self.input_dim = None
        self.hidden_dim = hidden_dim
        self.pretrain_epochs = pretrain_echops

    def pretrain_feature_extractor(self, X, Y):
        """
        Pre-train backbone feature_extractor trên task classification
        """
        # Chuyển X, Y thành TensorDataset
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.long)  # giả sử multi-class
        ds = TensorDataset(X_tensor, Y_tensor)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        # Tạo tạm thời classifier head
        feat_dim = self.hidden_dim if self.use_fe else self.input_dim
        classifier = nn.Sequential(
            self.feature_extractor,
            nn.Linear(feat_dim, 2)
        ).to(self.device)
        optim_clf = optim.Adam(classifier.parameters(), lr=self.lr)
        criterion_clf = nn.CrossEntropyLoss()

        for epoch in range(1, self.pretrain_epochs + 1):
            classifier.train()
            total_loss = 0
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optim_clf.zero_grad()
                logits = classifier(xb)
                loss = criterion_clf(logits, yb)
                loss.backward()
                optim_clf.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            print(f"[Pretrain] Epoch {epoch}/{self.pretrain_epochs} — Loss: {avg:.4f}")

        # Sau khi pretrain xong, giữ lại backbone
        # (classifier[0] chính là self.feature_extractor đã được training)
        self.feature_extractor = classifier[0]
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    
    def compute_kernel_matrix(self, X1, X2, weights, gamma, bias=0.0):
        """
        Tính ma trận kernel giữa X1 (n1 x d) và X2 (n2 x d):
            K(x,y) = sum_{m=1}^{num_kernels} w_m * exp(-gamma_m * ||x-y||^2) + bias
        """
        # Tính ||x||^2 của từng sample
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        # Tính khoảng cách bình phương: ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x^T*y
        dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        # Tính các kernel riêng lẻ cho mỗi gamma
        kernel_stack = np.array([np.exp(-g * dist_sq) for g in gamma])
        # Kết hợp các kernel với trọng số và cộng bias
        K = np.tensordot(weights, kernel_stack, axes=1) + bias
        return K
    
    # early stopping for validation accuracy
    def fit(self, Xs_train, Y_train, Xs_test=None, Y_test=None):
        self.X_train = Xs_train.copy()
        self.input_dim = Xs_train.shape[1]
        if self.use_fe:
            # print("Using feature extractor for training")
            self.feature_extractor = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
            )
            feat_dim = self.hidden_dim
        else:
            # identity mapping
            # print("Using identity mapping for feature extractor")
            self.feature_extractor = lambda x: x
            feat_dim = self.input_dim

        # Phase 1: Pre-train feature extractor if needed
        if self.pretrain_epochs > 0 and self.use_fe:
            print("Pre-training feature extractor...")
            self.pretrain_feature_extractor(Xs_train, Y_train)
            print("Pre-training completed.")

        # Phase 2: Finetune the whole network RBFNet
        # Tách training và validation set
        X_train, X_val, Y_train, Y_val = train_test_split(
            Xs_train, Y_train, test_size=0.2, random_state=42, stratify=Y_train
        )

        train_dataset = PairDataset(X_train, Y_train)
        val_dataset = PairDataset(X_val, Y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        self.model = RBFNet(feature_extractor= self.feature_extractor, num_kernels=self.num_kernels).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()

        best_misclf_rate = float('inf')
        best_val_loss = float('inf')
        
        trigger_times = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0
            for batch_x1, batch_x2, batch_labels in train_loader:
                batch_x1, batch_x2, batch_labels = batch_x1.float(), batch_x2.float(), batch_labels.float()
                batch_x1 = batch_x1.to(self.device)
                batch_x2 = batch_x2.to(self.device)
                batch_labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x1, batch_x2)
                loss = criterion(outputs, batch_labels)

                if self.lambda_reg != 0:
                    weights_rbf = self.model.log_weights_rbf
                    loss_l1 = self.lambda_reg * torch.sum(torch.abs(weights_rbf))
                    loss += loss_l1

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            # --- Validation Phase ---
            self.model.eval()
            correct = 0
            total = 0
            total_val_loss = 0
            with torch.no_grad():
                for val_x1, val_x2, val_labels in val_loader:
                    val_x1, val_x2 = val_x1.float().to(self.device), val_x2.float().to(self.device)
                    val_labels = val_labels.float().to(self.device)
                    outputs = self.model(val_x1, val_x2)
                    val_loss = criterion(outputs, val_labels)
                    total_val_loss += val_loss.item()

                    preds = (outputs > 0.5).float()
                    correct += (preds == val_labels).sum().item()
                    total += val_labels.size(0)

            avg_val_loss = total_val_loss / len(val_loader)
            acc = correct / total
            misclf_rate = 1.0 - acc

            print(f"Epoch {epoch}/{self.epochs}, Train Loss: {total_loss/len(train_loader):.4f},Val Loss: {avg_val_loss:.4f}, Validation Misclassification Rate: {misclf_rate:.4f}")

            # # --- Early Stopping ---
            # if misclf_rate < best_misclf_rate:
            #     best_misclf_rate = misclf_rate
            #     trigger_times = 0
            #     best_model_state = self.model.state_dict()
            # else:
            #     trigger_times += 1
            #     if trigger_times >= self.patience:
            #         print(f"Early stopping at epoch {epoch}")
            #         break
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                trigger_times = 0
                best_model_state = self.model.state_dict()
            else:
                trigger_times += 1
                if trigger_times >= self.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Load best model state
        self.model.load_state_dict(best_model_state)

        # Lưu lại tham số đã học
        self.model.eval()
        with torch.no_grad():
            self.learned_gamma = torch.exp(self.model.log_gamma).cpu().numpy()
            self.learned_weights = torch.exp(self.model.log_weights_rbf).data.cpu().numpy()
            self.learned_bias = self.model.bias.item()

        print("Training completed.")
        print("Learned gamma:", self.learned_gamma)
        print("Learned weights:", self.learned_weights)
        print("Learned bias:", self.learned_bias)


    def transform(self, X, compute_self_kernel=False):
        """
        Tính ma trận kernel giữa X và X_train với trọng số learned từ RBFNet.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input data to transform
        compute_self_kernel : bool, default=False
            If True, also compute and return the kernel matrix between X and itself
            
        Returns:
        --------
        K_train : numpy.ndarray
            Kernel matrix between X and X_train
        K_self : numpy.ndarray, optional
            Kernel matrix between X and X (only if compute_self_kernel=True)
        """
        if self.X_train is None:
            raise ValueError("Model has not been fitted yet. Please call fit() first.")
        if self.learned_gamma is None or self.learned_weights is None:
            raise ValueError("Model parameters not learned. Please call fit() first.")

        # Compute kernel matrix between X and X_train
        K_train = self.compute_kernel_matrix(X, self.X_train, self.learned_weights,
                                    self.learned_gamma, bias=self.learned_bias)
        
        # Optionally compute kernel matrix between X and itself (e.g., X_test and X_test)
        if compute_self_kernel:
            K_self = self.compute_kernel_matrix(X, X, self.learned_weights,
                                        self.learned_gamma, bias=self.learned_bias)
            return K_self
        
        return K_train


        

