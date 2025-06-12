import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from ..utils.pair_dataset import MultiPairDataset, multimodal_train_test_split, SmartMultiPairDataset


class RBFModule(nn.Module):
    """
    Module này gộp cả tính toán RBF kernel và tổng hợp (fusion) các kernel features 
    của một modality thành một scalar score.
    """
    def __init__(self, input_dim,feature_extractor, num_kernels=10):
        super(RBFModule, self).__init__()
        self.num_kernels = num_kernels
        # Tham số log_gamma đảm bảo gamma = exp(log_gamma) luôn > 0
        self.log_gamma = nn.Parameter(torch.randn(num_kernels))
        # Tham số weighted fusion cho các kernel features
        # self.weights = nn.Parameter(torch.empty(num_kernels))
        self.log_weights = nn.Parameter(torch.empty(num_kernels))
        nn.init.xavier_uniform_(self.log_weights.unsqueeze(0))
        self.bias = nn.Parameter(torch.zeros(1))
        self.feature_extractor = feature_extractor
        
    def forward(self, x1, x2):
        """
        x1, x2: tensor có shape [batch_size, input_dim]
        Trả về: tensor [batch_size] là raw logit của modality này
        """
        x1 = self.feature_extractor(x1)
        x2 = self.feature_extractor(x2)
        # Tính khoảng cách bình phương giữa x1 và x2: [batch_size, 1]
        diff = torch.sum((x1 - x2) ** 2, dim=1).unsqueeze(1)
        gamma = torch.exp(self.log_gamma)  # [num_kernels]
        phi_rbf = torch.exp(- gamma * diff)  # [batch_size, num_kernels]
        # Tính weighted sum của kernel features và cộng bias
        score = torch.matmul(phi_rbf, torch.exp(self.log_weights)) + self.bias  # [batch_size] nn.Linear(num_kernels, 1)
        return score


class MultiModal(nn.Module):
    """
    Mô hình similarity learning cho đa modality:
      - Mỗi modality được xử lý qua một RBFModule độc lập.
      - Các đầu ra (logits) của từng modality được gộp lại và kết hợp thông qua một lớp weighted fusion toàn cục.
      - Hàm forward trả về cả global fusion logits và danh sách logits của từng nhánh.
    """
    def __init__(self, modality_dims: dict, modality_fextractors: dict, modality_kernels: dict):
        super(MultiModal, self).__init__()
        self.modalities = list(modality_dims.keys())
        self.rbf_modules = nn.ModuleDict({
            modality: RBFModule(modality_dims[modality], modality_fextractors[modality], modality_kernels[modality])
            for modality in self.modalities
        })
        # Các tham số fusion global: trọng số cho từng modality và global bias
        # self.global_weights = nn.Parameter(torch.empty(len(self.modalities)))
        self.log_global_weights = nn.Parameter(torch.empty(len(self.modalities)))
        nn.init.xavier_uniform_(self.log_global_weights.unsqueeze(0))
        self.global_bias = nn.Parameter(torch.zeros(1))

    def forward(self, inputs1: dict, inputs2: dict):
        modality_logits = []
        modality_probs = []
        for modality in self.modalities:
            x1, x2 = inputs1[modality], inputs2[modality]
            # Lấy raw logit của từng modality (chưa qua sigmoid)
            logit = self.rbf_modules[modality](x1, x2)  
            modality_logits.append(logit)
            modality_prob = torch.sigmoid(logit)  # [batch_size]
            modality_probs.append(modality_prob)
            
        # Ghép các logits thành ma trận [batch_size, num_modalities]
        fused = torch.stack(modality_logits, dim=1)
        # global_weights = torch.sigmoid(self.log_global_weights)  # [num_modalities]
        global_weights = torch.softmax(self.log_global_weights, dim=0)  # [num_modalities]

        # Fusion global: weighted sum của các nhánh cộng global bias
        global_logits = torch.matmul(fused, global_weights.unsqueeze(1)).squeeze(1) + self.global_bias  
        # global_logits = torch.sigmoid(global_logits)  # [batch_size]
        global_probs = torch.sigmoid(global_logits)  # [batch_size]
        return global_probs, modality_probs 


# ------------------------------------------
# Lớp DeepMultiKernelLearning 
# ------------------------------------------
class DeepMultiKernelLearning():
    """
    Lớp Deep Kernel Learning (DKL) cho bài toán similarity learning với đa modality.
    
    Đầu vào Xs_train, Xs_test là dictionary với:
      - key: tên modality
      - value: numpy array có shape (n_samples, n_features)
    
    Hàm sample_multimodal_pairs tạo cặp dữ liệu dựa trên nhãn, sau đó hàm fit huấn luyện model.
    Hàm transform tính ma trận kernel giữa tập dữ liệu mới và tập train theo công thức:
      K(x, y) = Σ_{mod} [ Σ_{m=1}^{num_kernels_mod} w_m * exp(-gamma_m * ||x - y||^2) + bias_mod ]
                 * (trọng số global của modality) + global bias.
    """
    def __init__(self,
                 num_kernels=10, 
                 lr=0.001, epochs=10, 
                 pretrain_epochs=1000,
                 patience=10, 
                 batch_size:int =64, 
                 hidden_dim=64,
                 use_fe=False,
                 device=None):
        self.num_kernels = num_kernels  # Số kernel mặc định cho mỗi modality
        self.lr = lr
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs  # Số epoch pretrain cho từng modality
        self.batch_size = batch_size
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_train = None  # Lưu tập train (dạng dict) để dùng trong transform
        self.learned_bias = None  # Lưu global bias sau khi huấn luyện
        self.global_weights = None  # Lưu global weights sau khi huấn luyện
        self.patience = patience
        self.use_fe = use_fe
        self.input_dim = None
        self.hidden_dim = hidden_dim

    def pretrain_feature_extractors(self, Xs_train: dict, Y_train):
        """
        Pre-train riêng feature_extractor cho mỗi modality qua task phân lớp (2-class).
        Sau khi xong, extractor được giữ lại trong self.model.rbf_modules[mod].feature_extractor
        và được freeze (requires_grad=False).
        """
        modality_names = list(Xs_train.keys())
        num_classes = len(np.unique(Y_train))
        criterion = nn.CrossEntropyLoss()

        for mod in modality_names:
             # Bỏ qua CSF
            if mod == "CSF":
                print(f"→ Skipping pre-training for modality '{mod}' (no feature_extractor).")
                continue
            print(f"→ Pretraining feature_extractor for modality '{mod}'")

            # 1) Lấy feature_extractor của module
            module = self.model.rbf_modules[mod]
            fe = module.feature_extractor

            # 2) Tạm gắn head classification lên extractor
            hidden_dim = self.hidden_dim if self.use_fe else self.input_dim
            clf = nn.Sequential(
                fe,
                nn.Linear(hidden_dim, num_classes)
            ).to(self.device)
            optimizer = optim.Adam(clf.parameters(), lr=self.lr)

            # 3) Tạo DataLoader cho Xs_train[mod], Y_train
            X_mod = torch.tensor(Xs_train[mod], dtype=torch.float32)
            Y_mod = torch.tensor(Y_train,   dtype=torch.long)
            ds = TensorDataset(X_mod, Y_mod)
            loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

            # 4) Vòng pretrain
            for epoch in range(1, 501):
                clf.train()
                total_loss = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    optimizer.zero_grad()
                    logits = clf(xb)              # forward qua extractor + head
                    loss   = criterion(logits, yb)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                if epoch % 100 == 0 or epoch in (1, self.pretrain_epochs):
                    print(f"   [Modality={mod}] Epoch {epoch}/{self.pretrain_epochs} — Loss: {total_loss/len(loader):.4f}")

            # 5) Tách extractor ra và freeze
            module.feature_extractor = clf[0]
            for p in module.feature_extractor.parameters():
                p.requires_grad = False

            print(f"✔ Finished pre-training extractor for '{mod}'\n")



    def pretrain_modalities(self, Xs_train: dict, Y_train):
        """
        Pretrain từng nhánh (modality) riêng biệt.
        Sử dụng SmartMultiModalPairDataset để sinh cặp dữ liệu và huấn luyện độc lập.
        """
        modality_names = list(Xs_train.keys())
        print("Pretraining từng modality:")

        # Loss function
        criterion = nn.BCEWithLogitsLoss()

        # Tạo dataset một lần duy nhất
        smart_dataset = SmartMultiPairDataset(
            X_dict=Xs_train,
            y=Y_train,
            modality_names=modality_names,
            distance=False  # 1 = cùng lớp, 0 = khác lớp
        )
        dataloader = torch.utils.data.DataLoader(
            smart_dataset, batch_size=self.batch_size, shuffle=True
        )

        # Pretrain từng modality độc lập
        for mod in modality_names:
            print(f"  -> Pretraining modality: {mod}")
            optimizer = optim.Adam(self.model.rbf_modules[mod].parameters(), lr=self.lr)

            for epoch in range(1, self.pretrain_epochs + 1):
                self.model.rbf_modules[mod].train()
                total_loss = 0.0
                for batch in dataloader:
                    x1_dict, x2_dict, labels = batch
                    x1 = x1_dict[mod].to(self.device)
                    x2 = x2_dict[mod].to(self.device)
                    labels = labels.to(self.device)

                    optimizer.zero_grad()
                    logit = self.model.rbf_modules[mod](x1, x2)
                    loss = criterion(logit, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(dataloader)
                if epoch % 100 == 0 or epoch == 1 or epoch == self.pretrain_epochs:
                    print(f"      Epoch {epoch}/{self.pretrain_epochs}, Loss: {avg_loss:.4f}")

            print(f"  -> Modality {mod} pretraining final loss: {avg_loss:.4f}")
        print("✅ Pretraining các modality hoàn tất.\n")


    def fit(self, Xs_train: dict, Y_train, Xs_test=None, Y_test=None):
        """
        Huấn luyện mô hình:
          - Xs_train: dict, key là modality, value là numpy array [n_samples, modality_dim]
          - Y_train: numpy array [n_samples] chứa nhãn
        """
        
        # Lưu lại tập train để dùng trong transform
        self.X_train = {mod: Xs_train[mod].copy() for mod in Xs_train}
        modality_names = list(self.X_train.keys())
        print("Modality names:", modality_names)
        modality_dims = {mod: self.X_train[mod].shape[1] for mod in modality_names}
        # Đặt số kernel cho mỗi modality: ví dụ "CSF" dùng 6 kernels, các modality khác dùng self.num_kernels
        modality_kernels = {mod: 3 if mod == "CSF" else self.num_kernels for mod in modality_names}
        # modality_kernels = {mod: self.num_kernels for mod in modality_names}
            
        modality_fes = {}
        for mod in modality_names:
            if mod == "CSF":
                # CSF luôn không dùng feature_extractor
                modality_fes[mod] = nn.Identity().to(self.device)
            else:
                # Các modality khác: nếu use_fe thì tạo Sequential, không thì Identity
                if self.use_fe:
                    modality_fes[mod] = nn.Sequential(
                        nn.Linear(modality_dims[mod], self.hidden_dim),
                        nn.ReLU(),
                        nn.Linear(self.hidden_dim, self.hidden_dim),
                    ).to(self.device)
                else:
                    modality_fes[mod] = nn.Identity().to(self.device)


        Xs_train, Xs_val, Y_train, Y_val = multimodal_train_test_split(
            Xs_train, Y_train, test_size=0.15, random_state=42, stratify=Y_train
        )
        # train_dataset = MultiPairDataset(Xs_train, Y_train, modality_names)
        # val_dataset = MultiPairDataset(Xs_val, Y_val, modality_names)
        train_dataset = SmartMultiPairDataset(Xs_train, Y_train, modality_names)
        val_dataset = SmartMultiPairDataset(Xs_val, Y_val, modality_names)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        print("len(train_loader):", len(train_loader))
        # Khởi tạo model đa modality với các module đã được gộp
        self.model = MultiModal(modality_dims,modality_fes, modality_kernels).to(self.device)

        # --- Bước 0: Pretrain feature extractor ---
        self.pretrain_feature_extractors(Xs_train, Y_train)

        # --- Bước 1: Pretrain từng nhánh modality ---
        self.pretrain_modalities(Xs_train, Y_train)
        # --- Bước 2: Fine-tune toàn mô hình ---
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        # Sử dụng BCEWithLogitsLoss cho fine-tuning (với raw logits)
        criterion = nn.BCEWithLogitsLoss()

        best_misclf_rate = float('inf')
        best_val_loss = float('inf')
        trigger_times = 0
        best_model_state = None

        print("Fine-tuning toàn mô hình với composite loss:")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss, interaction_total, individual_total = 0.0, 0.0, 0.0
            num_batches = 0

            for batch in train_loader:
                batch_X1, batch_X2, batch_labels = batch

                inputs1 = {mod: batch_X1[mod].to(self.device) for mod in modality_names}
                inputs2 = {mod: batch_X2[mod].to(self.device) for mod in modality_names}
                labels = batch_labels.to(self.device)

                optimizer.zero_grad()
                global_logits, modality_logits = self.model(inputs1, inputs2)
                

                interaction_loss = criterion(global_logits, labels)
                individual_loss = torch.stack([
                    criterion(logit, labels) for logit in modality_logits
                ]).mean()

                loss = interaction_loss + individual_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                interaction_total += interaction_loss.item()
                individual_total += individual_loss.item()
                num_batches += 1

            avg_train_loss = total_loss / num_batches

            # --- Validation Phase ---
            self.model.eval()
            correct = 0
            total = 0
            total_val_loss, individual_val_loss, interaction_val_loss = 0.0, 0.0, 0.
            num_val_batches = 0
            with torch.no_grad():
                for val_X1, val_X2, val_labels in val_loader:
                    val_inputs1 = {mod: val_X1[mod].to(self.device) for mod in modality_names}
                    val_inputs2 = {mod: val_X2[mod].to(self.device) for mod in modality_names}
                    val_labels = val_labels.to(self.device)

                    val_logits, _ = self.model(val_inputs1, val_inputs2)
                    interaction_val_loss = criterion(val_logits, val_labels)
                    individual_val_loss = torch.stack([
                        criterion(logit, val_labels) for logit in _  # modality_logits
                    ]).mean()
                    individual_val_loss += individual_val_loss.item()
                    interaction_val_loss += interaction_val_loss.item()
                    total_val_loss += interaction_val_loss + individual_val_loss

                    preds = (torch.sigmoid(val_logits) > 0.5).float()
                    correct += (preds == val_labels).sum().item()
                    total += val_labels.size(0)
                    num_val_batches += 1
            avg_val_loss = total_val_loss/num_val_batches

            acc = correct / total
            misclf_rate = 1.0 - acc
            if epoch % 50 == 0 or epoch == 1 or epoch == self.epochs:
                print(f"[Epoch {epoch:03d}/{self.epochs}] Train Loss: {avg_train_loss:.4f},Val Loss: {avg_val_loss:.4f}, Val Misclf Rate: {misclf_rate:.4f}")

            # --- Early Stopping ---
            # if misclf_rate < best_misclf_rate:
            #     best_misclf_rate = misclf_rate
            #     best_model_state = self.model.state_dict()
            #     trigger_times = 0
            # else:
            #     trigger_times += 1
            #     print(f"  No improvement. EarlyStopping counter: {trigger_times}/{patience}")
            #     if trigger_times >= patience:
            #         print(f" Early stopping at epoch {epoch}")
            #         break
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                trigger_times = 0
            else:
                trigger_times += 1
                # print(f"  No improvement. EarlyStopping counter: {trigger_times}/{patience}")
                if trigger_times >= self.patience:
                    print(f" Early stopping at epoch {epoch}")
                    break
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            # print(f"Best model loaded (misclf_rate ={best_misclf_rate:.4f})")
            print(f"Best model loaded (val_loss ={best_val_loss:.4f})")

        # Lưu lại các tham số đã học (nếu dùng RBFNet bên trong)
        # Sau khi huấn luyện, lưu lại global bias từ mô hình
        self.model.eval()
        with torch.no_grad():
            self.learned_bias = self.model.global_bias.item()
            # self.global_weights = torch.sigmoid(self.model.log_global_weights).detach().cpu().numpy()
            self.global_weights = torch.softmax(self.model.log_global_weights, dim=0).detach().cpu().numpy()
        print("Fine-tuning hoàn tất.")
        print("Learned global bias:", self.learned_bias)
        print("Global weights:", self.global_weights)



    
    def compute_kernel_matrix_mod(self, X1, X2, weights, gamma, bias=0.0):
        """
        Tính ma trận kernel cho một modality với công thức:
          K_mod(x, y) = sum_{m=1}^{num_kernels_mod} w_m * exp(-gamma_m * ||x-y||^2) + bias
        - X1: numpy array [n1, d]
        - X2: numpy array [n2, d]
        - weights: vector (num_kernels_mod,)
        - gamma: vector (num_kernels_mod,)
        Trả về: ma trận K_mod có shape [n1, n2]
        """
        X1_sq = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2 ** 2, axis=1).reshape(1, -1)
        dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
        kernel_stack = np.array([np.exp(-g * dist_sq) for g in gamma])
        K_mod = np.tensordot(weights, kernel_stack, axes=1)
        K_mod = K_mod + bias
        return K_mod

    def transform(self, Xs: dict):
        """
        Tính ma trận kernel giữa tập dữ liệu mới Xs và tập train theo công thức:
          K(x, y) = sum_{mod in modalities} [ sum_{m=1}^{num_kernels_mod} w_m * exp(-gamma_m * ||x-y||^2) + bias_mod ]
                    * (trọng số global của modality) + global bias.
                    
        Các tham số (gamma, weights và bias) của mỗi modality được lấy từ model.rbf_modules[mod].
        """
        if self.X_train is None:
            raise ValueError("Model chưa được huấn luyện. Vui lòng gọi hàm fit() trước.")
            
        modality_names = list(self.X_train.keys())
        K_total = None
        
        for mod in modality_names:
            rbf_module = self.model.rbf_modules[mod]
            gamma_mod = torch.exp(rbf_module.log_gamma).detach().cpu().numpy()  # [num_kernels_mod]
            weights_mod = torch.exp(rbf_module.log_weights).detach().cpu().numpy()             # [num_kernels_mod]
            bias_mod = rbf_module.bias.detach().cpu().numpy().item()
            
            X_train_mod = self.X_train[mod]  # [n_train, d]
            X_mod = Xs[mod]                  # [n_test, d]
            
            X_mod_sq = np.sum(X_mod ** 2, axis=1).reshape(-1, 1)
            X_train_sq = np.sum(X_train_mod ** 2, axis=1).reshape(1, -1)
            dist_sq = X_mod_sq + X_train_sq - 2 * np.dot(X_mod, X_train_mod.T)
            
            kernel_stack = np.array([np.exp(-g * dist_sq) for g in gamma_mod])
            K_mod = np.tensordot(weights_mod, kernel_stack, axes=1)
            K_mod = K_mod + bias_mod
            
            mod_index = self.model.modalities.index(mod)
            # global_w = torch.sigmoid(self.model.log_global_weights).detach().cpu().numpy()[mod_index]
            global_w = torch.softmax(self.model.log_global_weights, dim=0).detach().cpu().numpy()[mod_index]
            K_mod = global_w * K_mod
            
            if K_total is None:
                K_total = K_mod
            else:
                K_total += K_mod
        
        global_bias = self.model.global_bias.detach().cpu().numpy().item()
        K_total = K_total + global_bias
        return K_total
