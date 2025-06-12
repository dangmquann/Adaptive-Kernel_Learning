import numpy as np
import matplotlib.pyplot as plt
import os

def visualize_sorted_kernel(kernel_matrix,args , labels, version, modality, normalize_kernel_fn):
    """
    Chuẩn hóa, sắp xếp và trực quan hóa ma trận kernel theo nhãn.

    Args:
        kernel_matrix (np.ndarray): Ma trận kernel đầu vào (n x n).
        labels (np.ndarray): Mảng nhãn tương ứng với từng dòng của kernel.
        version (str): Thư mục phiên bản để lưu hình ảnh.
        modality (str): Tên modality để đặt tên file ảnh.
        normalize_kernel_fn (callable): Hàm chuẩn hóa kernel, nhận đầu vào là np.ndarray.
    """
    # Bước 1: Chuẩn hóa
    normalized_kernel = normalize_kernel_fn(kernel_matrix)

    # Bước 2: Sắp xếp theo nhãn
    sorted_indices = np.argsort(labels)
    sorted_kernel = normalized_kernel[sorted_indices, :][:, sorted_indices]

    # Bước 3: Vẽ và lưu hình ảnh
    plt.figure(figsize=(10, 8))
    plt.imshow(sorted_kernel, cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    plt.axis('off')
    plt.show()
    os.makedirs(f'../../figures/{args.dataset}/{version}', exist_ok=True)
    plt.savefig(f'../../figures/{args.dataset}/{version}/{modality}.png')
    plt.close()

def normalize_kernel(K):
    # Cách 1: Chuẩn hoá về [0,1]
    K_min = np.min(K)
    K_max = np.max(K)
    return (K - K_min) / (K_max - K_min + 1e-8)

if __name__ == "__main__":
    # Example usage
    kernel_matrix = np.random.rand(10, 10)
    labels = np.random.randint(0, 2, size=10)
    args = type('', (), {})()  # Create a simple object to hold attributes
    args.dataset = 'example_dataset'
    version = 'v1'
    modality = 'example_modality'
    
    visualize_sorted_kernel(kernel_matrix, args, labels, version, modality, normalize_kernel)