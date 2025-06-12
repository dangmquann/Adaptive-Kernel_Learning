import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel, cosine_similarity

class EarlyKernel:
    def __init__(self, kernel_type="linear", **kwargs):
        """
        Initialize the EarlyKernel class.

        Args:
            kernel_type (str): The type of kernel to use (default: "rbf").
            **kwargs: Additional parameters for the kernel function.
        """
        self.kernel_type = kernel_type
        self.params = kwargs
        self.X_fit = None

    def get_kernel(self, X1, X2):
        """
        Compute the kernel matrix based on the specified kernel type.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: The computed kernel matrix.

        Raises:
            ValueError: If an unsupported kernel type is specified.
        """
        if self.kernel_type == "rbf":
            gamma = self.params.get("gamma", 0.5)
            return rbf_kernel(X1, X2, gamma=gamma)
        elif self.kernel_type == "linear":
            return linear_kernel(X1, X2)
        elif self.kernel_type == "polynomial":
            degree=self.params.get("degree", 2)
            coef0=self.params.get("coef0", 1)
            gamma=self.params.get("gamma", 3)
            return polynomial_kernel(X1, X2, degree=degree, coef0=coef0, gamma=gamma)
        elif self.kernel_type == "ensemble":
            degree=self.params.get("degree", 3)
            coef0=self.params.get("coef0", 1)
            gamma=self.params.get("gamma", 1.0)
            kernel_linear = linear_kernel(X1, X2)
            kernel_rbf = rbf_kernel(X1, X2)
            kernel_poly = polynomial_kernel(X1, X2, degree=degree, coef0=coef0, gamma=gamma)
            kernel_rbf1 = rbf_kernel(X1, X2, gamma=0.5)
            kernel_poly1 = polynomial_kernel(X1, X2, degree=2, coef0=1, gamma=3)
            kernels = np.array([kernel_linear, kernel_rbf, kernel_poly, kernel_rbf1, kernel_poly1])
            return kernels.mean(axis=0)
        elif self.kernel_type == "cosine":
            return cosine_similarity(X1, X2)
        else:
            raise ValueError(f"Unsupported kernel type: {self.kernel_type}")

    def fit(self, X, y=None):
        """
        Fit the kernel model with the provided data.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray, optional): Target labels (not used for unsupervised kernels).
        """
        self.X_fit = X
        return self

    def transform(self, X):
        """
        Transform the input data to kernel space using the fitted data.

        Args:
            X (np.ndarray): Feature matrix to transform.

        Returns:
            np.ndarray: Kernel matrix between the fitted data and the input data.

        Raises:
            ValueError: If the model has not been fitted.
        """
        if self.X_fit is None:
            raise ValueError("The kernel model must be fitted before calling transform.")
        return self.get_kernel(self.X_fit, X), self.get_kernel(X, X)

    def fit_transform(self, X, y=None):
        """
        Fit the kernel model and transform the input data to kernel space.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray, optional): Target labels (not used for unsupervised kernels).

        Returns:
            np.ndarray: Kernel matrix computed from the input data.
        """
        self.fit(X, y)
        return self.transform(X)

    def __str__(self):
        return f"EarlyKernel(kernel_type={self.kernel_type})"
