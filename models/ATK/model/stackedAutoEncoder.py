import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
import copy

def fro_norm(A):
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A)
    return torch.norm(A, p='fro')

class Autoencoder(nn.Module):
    """
    Mô hình autoencoder cơ bản:
      - Encoder: nén dữ liệu đầu vào thành latent vector Z
      - Decoder: tái tạo lại dữ liệu từ Z
    """
    def __init__(self, input_dim, hidden_dims, latent_dim, activation=nn.ReLU(), dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        self.encoder = self._build_encoder(input_dim, hidden_dims, latent_dim, activation, dropout_rate)
        self.decoder = self._build_decoder(input_dim, hidden_dims, latent_dim, activation, dropout_rate)

    def _build_encoder(self, input_dim, hidden_dims, latent_dim, activation, dropout_rate):
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, latent_dim))
        return nn.Sequential(*layers)

    def _build_decoder(self, input_dim, hidden_dims, latent_dim, activation, dropout_rate):
        layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(activation)
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, input_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

class ShallowAutoencoder(nn.Module):
    def __init__(self, input_dim, output_dim, activation=nn.ReLU()):
        super(ShallowAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            activation
        )
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, input_dim),
            activation
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def pretrain(self, X, epochs=10, batch_size=32, lr=1e-3, device=torch.device("cpu"), lmbda=0.75, gamma=1.0):
        """
        Huấn luyện shallow AE với loss kết hợp giữa reconstruction loss và code loss.
        lmbda: trọng số của code loss.
        gamma: tham số gamma cho RBF kernel tính P.
        """
        dataset = TensorDataset(X, X)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)
        self.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(device)
                optimizer.zero_grad()
                x_hat = self(batch_X)
                # Tính reconstruction loss
                recon_loss = criterion(x_hat, batch_X)
                # Tính code loss dựa trên đầu ra của encoder
                z = self.encoder(batch_X)
                P = rbf_torch(batch_X, gamma=gamma)
                C = linear_torch(z)
                c_loss = code_loss(C, P)
                loss = (1 - lmbda) * recon_loss + lmbda * c_loss
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(dataloader.dataset)
            print(f"  Shallow AE pretrain Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}")

def code_loss(C, P):#Z
    """
    Tính code loss theo chuẩn Frobenius.
    Z: (batch_size, latent_dim) hoặc (n_samples, latent_dim)
    P: (n_samples, n_samples) - ma trận kernel cho trước
    Return:
      Một scalar tensor tương ứng với ||C/||C||F - P/||P||F||_F
      với C = Z * Z^T.
    """
    # C = torch.matmul(Z, Z.t())
    C_norm = C / fro_norm(C).clamp_min(1e-12)
    P_norm = P / fro_norm(P).clamp_min(1e-12)
    return fro_norm(C_norm - P_norm)

def rbf_torch(X, gamma=1.0):
    sq_dists = torch.cdist(X, X, p=2)**2
    return torch.exp(-gamma * sq_dists)

def linear_torch(X):
    return torch.matmul(X, X.t())

def polynomial_torch(X, degree=3, gamma=None, coef0=1):
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return (gamma * torch.matmul(X, X.t()) + coef0)**degree

class MiddleKernel:
    """
    Sử dụng Autoencoder để trích xuất latent features và tính kernel dựa trên latent features.
    """
    def __init__(
        self,
        kernel_type="rbf",
        latent_dim=30,
        hidden_dims=[1000,60],
        activation=nn.ReLU(),
        epochs=1000,
        batch_size=64,
        lr=1e-4,
        lambda_=0.75,
        dropout_rate=0.1,
        patience=200,  # Số epoch không cải thiện trước khi dừng sớm
        device=None,
        pretrain_epochs=100,
        **kwargs
    ):
        self.lambda_ = lambda_
        self.kernel_type = kernel_type
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.dropout_rate = dropout_rate
        self.params = kwargs
        self.patience = patience

        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.input_dim = None
        self.X_fit = None

        self.pretrain_epochs = pretrain_epochs

    def fit(self, X, y=None):
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        # # Tính prior kernel matrix P sử dụng rbf
        # P = rbf_torch(X, gamma=1.0)

        self.input_dim = X.shape[1]
        self.X_original = X.clone().to(self.device)
        self.y_original = y


        self.model = Autoencoder(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            latent_dim=self.latent_dim,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        ).to(self.device)



        # ------------------------
        # Phase 1: Tiền huấn luyện từng lớp (layer-wise pretraining)
        # ------------------------
        print("Start layer-wise pretraining (Phase 1)")
        # Xác định chuỗi kích thước: [input_dim] + hidden_dims + [latent_dim]
        dims = [self.input_dim] + self.hidden_dims + [self.latent_dim]
        X_current = X.clone().to(self.device)
        num_layers = len(dims) - 1
        for i in range(num_layers):
            print(f"Pretraining layer {i+1}/{num_layers}: {dims[i]} -> {dims[i+1]}")
            # Tạo shallow AE cho layer hiện tại và tiền huấn luyện với loss kết hợp (reconstruction + code loss)
            shallow_ae = ShallowAutoencoder(input_dim=dims[i], output_dim=dims[i+1], activation=self.activation).to(self.device)
            shallow_ae.pretrain(
                X_current,
                epochs=self.pretrain_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                device=self.device,
                lmbda=self.lambda_,
                gamma=self.params.get("gamma", 1.0)
            )
            # Cập nhật trọng số của layer encoder tương ứng trong mô hình chính
            # Cấu trúc encoder của mô hình chính được xây dựng theo block: [Linear, activation, Dropout]
            if i < len(self.hidden_dims):
                layer_idx = i * 3  # vị trí của lớp Linear trong block
            else:
                layer_idx = len(self.hidden_dims) * 3  # lớp cuối (latent layer)
            self.model.encoder[layer_idx].weight.data = shallow_ae.encoder[0].weight.data.clone()
            self.model.encoder[layer_idx].bias.data = shallow_ae.encoder[0].bias.data.clone()
            # Cập nhật dữ liệu đầu ra cho layer tiếp theo
            self.model.encoder[layer_idx].eval()
            with torch.no_grad():
                X_current = self.model.encoder[layer_idx](X_current)
            print(f"  Output dimension after layer {i+1}: {X_current.shape[1]}")



        dataset = TensorDataset(X, X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True) # for X
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)

        finetune_epochs = self.epochs - self.pretrain_epochs
        min_delta = 1e-4
        best_loss = float('inf')
        trigger = 0
        best_model_state = None

        print("Start fine-tuning (Phase 2) with reconstruction + code loss")
        self.model.train()
        for epoch in range(finetune_epochs):
            epoch_loss = 0.0
            for batch_X, _ in dataloader:
                batch_X = batch_X.to(self.device)
                optimizer.zero_grad()
                X_recon = self.model(batch_X)
                recon_loss = criterion(X_recon, batch_X)

                # # Tính code loss trên toàn bộ dữ liệu (có thể tính theo batch nếu dữ liệu lớn)
                Z = self.model.encode(batch_X)
                # Z =self.model.encoder(X)

                # P = rbf_kernel(batch_X.detach().cpu().numpy(), batch_X.detach().cpu().numpy(), gamma=self.params.get("gamma", 1.0))
                # P = torch.from_numpy(P).float().to(self.device)
                P = rbf_torch(batch_X, gamma=self.params.get("gamma", 1.0))
                # P = linear_torch(batch_X)
                # P = polynomial_torch(batch_X, degree=3, gamma=1.0, coef0=1)
    
                C = linear_torch(Z)
                # C = rbf_torch(Z, gamma=1.0)
                # C = polynomial_torch(Z, degree=3, gamma=1.0, coef0=1)
                # C = torch.matmul(Z, Z.t())

                c_loss = code_loss(C, P)
                # # Z = self.model.encode(X.to(self.device))             
                # print("recon_loss = ",recon_loss)
                # print("c_loss = ",c_loss)

                total_loss = (1 - self.lambda_) * recon_loss + self.lambda_ * c_loss
                total_loss.backward()
                optimizer.step()
                epoch_loss += total_loss.item() * batch_X.size(0)

            epoch_loss /= len(dataloader.dataset)
            scheduler.step(epoch_loss)
            print(f"Fine-tuning Epoch [{epoch+1}/{self.epochs}], Loss: {epoch_loss:.4f}")

            # Early Stopping
            if best_loss - epoch_loss > min_delta:
                best_loss = epoch_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                trigger = 0
                print("Significant improvement, reset patience counter.")
            else:
                trigger += 1
                if trigger >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        print("Finish training... for creating latent features")

        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            Z = self.model.encode(X)
        self.X_fit = Z.cpu().numpy()
        self.X_original = self.X_original.cpu().numpy()
        return self

    def transform(self, X, y=None):  
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        self.model.eval()
        with torch.no_grad():
            X = X.to(self.device)
            X_latent = self.model.encode(X)
        if self.X_fit is None:
            raise ValueError("The kernel model must be fitted before calling transform.")
        
        latent_kernel_train = self.get_kernel(self.X_fit, X_latent.cpu().numpy())
        latent_kernel_test = self.get_kernel(X_latent.cpu().numpy(), X_latent.cpu().numpy())
        print("Shape of latent_kernel_train: ", latent_kernel_train.shape)

        original_kernel_train = self.get_kernel(self.X_original, X.cpu().numpy())
        print("Shape of original_kernel_train: ", original_kernel_train.shape)
        original_kernel_test = self.get_kernel(X.cpu().numpy(), X.cpu().numpy())
       
        return self.get_kernel(self.X_fit, X_latent.cpu().numpy()), self.get_kernel(X_latent.cpu().numpy(), X_latent.cpu().numpy())

    def fit_transform(self, X, y=None):
        print("check of y before fit", y)
        self.fit(X, y)
        print("check of y", y)
        return self.transform(X, y)

    def get_kernel(self, X1, X2):
        if self.kernel_type == "rbf":
            gamma = self.params.get("gamma", 1)
            return rbf_kernel(X1, X2, gamma=gamma)
        elif self.kernel_type == "linear":
            return linear_kernel(X1, X2)
        elif self.kernel_type == "poly":
            degree = self.params.get("degree", 3)
            coef0 = self.params.get("coef0", 1)
            gamma = self.params.get("gamma", None)
            return polynomial_kernel(X1, X2, degree=degree, gamma=gamma, coef0=coef0)
        elif self.kernel_type == "ensemble":
            kernel_linear = linear_kernel(X1, X2)
            kernel_rbf = rbf_kernel(X1, X2)
            kernel_poly = polynomial_kernel(X1, X2)
            kernels = np.array([kernel_linear, kernel_rbf, kernel_poly])
            return kernels.mean(axis=0)
            
            

        else:
            raise ValueError(f"Unsupported kernel_type: {self.kernel_type}")

    def __str__(self):
        return (f"MiddleKernel(kernel_type={self.kernel_type}, latent_dim={self.latent_dim}, "
                f"hidden_dims={self.hidden_dims}, params={self.params})")



if __name__ == "__main__":
    # Ví dụ sử dụng MiddleKernel với dữ liệu ngẫu nhiên
    X = np.random.rand(100, 20).astype(np.float32)

    mk = MiddleKernel(
        kernel_type="rbf",
        latent_dim=32,
        hidden_dims=[256,128,64],
        activation=nn.ReLU(),
        epochs=50,
        batch_size=16,
        lr=1e-3,
        lambda_=0.75,
        dropout_rate=0.2,
        gamma=0.5,
        patience=10
    )

    mk.fit(X)
    latent_kernel, self_kernel = mk.transform(X)
    print("Latent kernel shape:", latent_kernel.shape)
    print("Self kernel shape:", self_kernel.shape)