import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import random


class PairDataset(Dataset):
    def __init__(self, X, y):
        """
        X: numpy array (N, d) - d là số chiều dữ liệu
        y: numpy array (N,)   - nhãn tương ứng
        """
        self.X = X
        self.y = y
        self.pairs = []
        self.labels = []
        
        # Gom index theo từng lớp
        idx_by_class = {}
        for idx, label in enumerate(y):
            idx_by_class.setdefault(label, []).append(idx)

        unique_labels = list(idx_by_class.keys())
        
        # Tạo positive pairs
        for lab in unique_labels:
            idxs = idx_by_class[lab]
            # Tạo mọi cặp (hoặc một subset) trong lớp lab
            for i in range(len(idxs)):
                for j in range(i+1, len(idxs)):
                    self.pairs.append((idxs[i], idxs[j]))
                    self.labels.append(1)
        
        # Tạo negative pairs
        for i in range(len(unique_labels)):
            for j in range(i+1, len(unique_labels)):
                idxs_i = idx_by_class[unique_labels[i]]
                idxs_j = idx_by_class[unique_labels[j]]
                for ii in idxs_i:
                    for jj in idxs_j:
                        self.pairs.append((ii, jj))
                        self.labels.append(0)
        
        # Chuyển sang numpy array
        self.pairs = np.array(self.pairs)
        self.labels = np.array(self.labels, dtype=np.float32)
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        i1, i2 = self.pairs[idx]
        x1 = self.X[i1]
        x2 = self.X[i2]
        label = self.labels[idx]
        return x1, x2, label


class MultiPairDataset(Dataset):
    def __init__(self, X: dict, y: np.ndarray, modality_names: list):
        """
        Khởi tạo dataset sinh cặp đầy đủ cho multi-modal input.
        
        - X: dict[modality] -> numpy array (n_samples, dim_modality)
        - y: numpy array (n_samples,), là nhãn lớp (int)
        - modality_names: list tên các modality, ví dụ ['MRI', 'PET', 'CSF']
        """
        self.X = X
        self.y = y
        self.modality_names = modality_names
        self.pairs = []
        self.labels = []

        n_samples = len(y)
        
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                self.pairs.append((i, j))
                label = 1 if y[i] == y[j] else 0
                self.labels.append(label)

        self.labels = np.array(self.labels, dtype=np.float32)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2 = self.pairs[idx]
        data1 = {}
        data2 = {}
        for modality in self.modality_names:
            data1[modality] = torch.tensor(self.X[modality][i1], dtype=torch.float32)
            data2[modality] = torch.tensor(self.X[modality][i2], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return data1, data2, label


class SmartMultiPairDataset(Dataset):
    def __init__(self, X_dict: dict, y: np.ndarray, modality_names: list, distance=False):
        """
        Sinh dataset chứa các cặp (positive/negative) theo cách smart pairing.

        - X_dict: dict[str, np.ndarray] - mỗi modality chứa [n_samples, d]
        - y: np.ndarray [n_samples, num_classes] - nhãn dạng one-hot
        - modality_names: danh sách tên modality
        - distance: nếu True thì dùng loss theo khoảng cách (0: cùng lớp, 1: khác lớp)
        """
        self.modality_names = modality_names
        self.X_dict = X_dict
        self.y = y
        self.distance = distance

        # Convert one-hot → class index
        num_classes = 2
        lb = LabelBinarizer()
        lb.fit(range(num_classes))
        class_ids = lb.inverse_transform(y)

        # Lưu index từng lớp
        digit_indices = [np.where(class_ids == i)[0] for i in range(num_classes)]
        class_counter = {
            i: {"counter": 0, "max": len(digit_indices[i])}
            for i in range(num_classes)
        }

        self.X1 = {mod: [] for mod in modality_names}
        self.X2 = {mod: [] for mod in modality_names}
        self.labels = []
        self.targets1 = []
        self.targets2 = []

        positive = True
        for i in range(len(y)):
            class_i = class_ids[i]

            if positive:
                # Positive: lấy từ cùng lớp
                new_idx = self._get_same_class_idx(class_i, digit_indices, class_counter)
                label = 0 if distance else 1
            else:
                # Negative: lấy từ khác lớp
                new_idx = self._get_other_class_idx(class_i, num_classes, digit_indices, class_counter)
                label = 1 if distance else 0

            # Gán từng modality
            for mod in modality_names:
                self.X1[mod].append(X_dict[mod][i])
                self.X2[mod].append(X_dict[mod][new_idx])

            self.labels.append(label)
            self.targets1.append(y[i])
            self.targets2.append(y[new_idx])
            positive = not positive  # toggle

        # Convert sang tensor
        for mod in modality_names:
            self.X1[mod] = torch.tensor(np.stack(self.X1[mod]), dtype=torch.float32)
            self.X2[mod] = torch.tensor(np.stack(self.X2[mod]), dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.targets1 = torch.tensor(np.stack(self.targets1), dtype=torch.float32)
        self.targets2 = torch.tensor(np.stack(self.targets2), dtype=torch.float32)

    def _get_same_class_idx(self, class_id, indices, counter_dict):
        second_idx = counter_dict[class_id]["counter"]
        max_idx = counter_dict[class_id]["max"]
        counter_dict[class_id]["counter"] = (second_idx + 1) % max_idx
        return indices[class_id][second_idx]

    def _get_other_class_idx(self, this_class, num_classes, indices, counter_dict):
        inc = random.randrange(1, num_classes)
        other_class = (this_class + inc) % num_classes
        second_idx = counter_dict[other_class]["counter"]
        max_idx = counter_dict[other_class]["max"]
        counter_dict[other_class]["counter"] = (second_idx + 1) % max_idx
        return indices[other_class][second_idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x1 = {mod: self.X1[mod][idx] for mod in self.modality_names}
        x2 = {mod: self.X2[mod][idx] for mod in self.modality_names}
        label = self.labels[idx]
        return x1, x2, label


def multimodal_train_test_split(Xs_dict, Y, test_size=0.2, random_state=42, stratify=None, return_indices=False):
    """
    Chia tập dữ liệu đa modality thành train/val.
    
    - Xs_dict: dict[str, np.ndarray] (modality -> [n_samples, d])
    - Y: np.ndarray [n_samples]
    - stratify: mặc định là Y (dùng để chia đều class)
    
    Trả về:
      - Xs_train: dict[modality] -> [n_train, d]
      - Xs_val: dict[modality] -> [n_val, d]
      - Y_train: [n_train]
      - Y_val: [n_val]
    """
    if stratify is None:
        stratify = Y

    indices = np.arange(len(Y))
    train_idx, val_idx, Y_train, Y_val = train_test_split(
        indices, Y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    Xs_train = {modality: Xs_dict[modality][train_idx] for modality in Xs_dict}
    Xs_val = {modality: Xs_dict[modality][val_idx] for modality in Xs_dict}
    
    if return_indices:
        return train_idx, val_idx, Xs_train, Xs_val, Y_train, Y_val
    
    return Xs_train, Xs_val, Y_train, Y_val


def sample_multimodal_pairs(X: dict, y, batch_size, modality_names):
        """
        Tạo batch các cặp dữ liệu đa modality:
          - X: dict, key là modality, value là numpy array [n_samples, modality_dim]
          - y: numpy array [n_samples] chứa nhãn
        Trả về:
          - inputs1, inputs2: dict với key là modality, value là numpy array [batch_size, modality_dim]
          - labels: numpy array [batch_size]
        """
        X1_dict = {modality: [] for modality in modality_names}
        X2_dict = {modality: [] for modality in modality_names}
        labels = []
        idx_by_class = {}
        for idx, label in enumerate(y):
            idx_by_class.setdefault(label, []).append(idx)
        half_bs = batch_size // 2
        # Positive pairs: cùng lớp
        for _ in range(half_bs):
            label = random.choice(list(idx_by_class.keys()))
            idxs = random.sample(idx_by_class[label], 2)
            for modality in modality_names:
                X1_dict[modality].append(X[modality][idxs[0]])
                X2_dict[modality].append(X[modality][idxs[1]])
            labels.append(1)
        # Negative pairs: khác lớp
        for _ in range(batch_size - half_bs):
            label1, label2 = random.sample(list(idx_by_class.keys()), 2)
            idx1 = random.choice(idx_by_class[label1])
            idx2 = random.choice(idx_by_class[label2])
            for modality in modality_names:
                X1_dict[modality].append(X[modality][idx1])
                X2_dict[modality].append(X[modality][idx2])
            labels.append(0)
        # Chuyển đổi sang numpy array
        for modality in modality_names:
            X1_dict[modality] = np.array(X1_dict[modality], dtype=np.float32)
            X2_dict[modality] = np.array(X2_dict[modality], dtype=np.float32)
        labels = np.array(labels, dtype=np.float32)
        return X1_dict, X2_dict, labels
    