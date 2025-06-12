import numpy as np


def normalize_kernel(K):
    K_min = np.min(K)
    K_max = np.max(K)
    return (K - K_min) / (K_max - K_min + 1e-8)




def stack_kernel_matrix_from_dict(Xs_kernel_dict):
    """
    Stacks all N x N matrices from a dictionary into a 3-way tensor.

    Parameters:
        Xs_kernel_dict (dict): A dictionary where values are NxN numpy arrays.

    Returns:
        numpy.ndarray: A 3D tensor (N x N x M), where M is the number of keys in the dictionary.
    """
    # Collect all the matrices into a list
    matrices = [kernel for kernel in Xs_kernel_dict.values()]

    # Stack them along a new dimension (the third axis)
    stacked_tensor = np.stack(matrices, axis=0)  # Shape will be (N, N)

    return stacked_tensor