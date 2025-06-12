import os
import numpy as np
from models.ATK.model import ShallowKernelLearning, DeepMultiKernelLearning, DeepKernelLearning
from MKLpy.algorithms import AverageMKL, EasyMKL

class KernelCombination:
    def __init__(self, method="sum", num_kernels=10, 
                 lr=0.01, epochs=1500, pretrain_epochs=500,
                 batch_size=64,patience=10, lambda_reg=None,
                 use_fe=False):
        self.method = method
        self.combined_kernel = None
        self.model = None
        # Truyền tham số cho DeepMultiKernelLearning
        self.num_kernels = num_kernels
        self.lr = lr
        self.epochs = epochs
        self.pretrain_epochs = pretrain_epochs
        self.batch_size = batch_size
        self.lambda_reg = lambda_reg
        self.patience = patience
        self.use_fe = use_fe
    def fit(self, KX_train, KY_train, KX_train_test, KX_test_test):
        if self.method == "sum":
            pass
        elif self.method == "SKL":
            self.model = ShallowKernelLearning(num_kernels=self.num_kernels, 
                                                 lr=self.lr, 
                                                 epochs=self.epochs, pretrain_echops=self.pretrain_epochs,
                                                 batch_size=self.batch_size,
                                                 lambda_reg=self.lambda_reg,
                                                 patience=self.patience,
                                                 use_fe=self.use_fe)
            self.model.fit(KX_train, KY_train, KX_train_test, KX_test_test)
        elif self.method == "AMKL":
            self.model = DeepMultiKernelLearning(num_kernels=self.num_kernels, 
                                                 lr=self.lr, 
                                                 epochs=self.epochs,pretrain_epochs=self.pretrain_epochs,
                                                 patience=self.patience,
                                                 batch_size=self.batch_size,
                                                 use_fe=self.use_fe)
            self.model.fit(KX_train, KY_train, KX_train_test, KX_test_test)
        elif self.method == "EasyMKL":
            self.model = EasyMKL(lam=0.8)
            self.model.fit(KX_train, KY_train)
        elif self.method == "AverageMKL":
            self.model = AverageMKL()
            self.model.fit(KX_train, KY_train)
        elif self.method == "DKL":
            self.model = DeepKernelLearning(input_dim=KX_train.shape[1],use_feature_extractor=False,epochs=self.epochs)
            self.model.fit(KX_train, KY_train)
        else:
            raise ValueError("Unknown kernel combination method")
        return self

    def transform(self, KX, compute_self_kernel=False):
        if self.method == "sum":
            return np.sum(KX,axis=0)
        elif self.method == "SKL":
            return self.model.transform(KX, compute_self_kernel)
        elif self.method == "AMKL":
            return self.model.transform(KX)
        elif self.method == "MHKF":
            return self.model.transform(KX)
        elif self.method == "EasyMKL":
            weights = self.model.solution.weights  # assumed shape: (n_kernels,)
            print("Weights: ", weights)
            # Convert weights to numpy array if it's not already
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)
            KX = [np.asarray(K, dtype=np.float64) for K in KX]  # ensure numeric arrays
            combined_kernel = np.zeros_like(KX[0], dtype=np.float64)  # initialize correctly

            for w, K in zip(weights, KX):
                combined_kernel += w * K

            return combined_kernel
        
        elif self.method == "AverageMKL":
            weights = self.model.solution.weights  # assumed shape: (n_kernels,)
            # Convert weights to numpy array if it's not already
            if not isinstance(weights, np.ndarray):
                weights = np.array(weights)
            KX = [np.asarray(K, dtype=np.float64) for K in KX]  # ensure numeric arrays
            combined_kernel = np.zeros_like(KX[0], dtype=np.float64)  # initialize correctly

            for w, K in zip(weights, KX):
                combined_kernel += w * K

            return combined_kernel

        elif self.method == "DKL":
            return self.model.transform(KX)

        else:
            raise ValueError("Unknown kernel combination method")