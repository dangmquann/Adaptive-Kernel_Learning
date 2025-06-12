from models.kernels.early_kernel import EarlyKernel
# from .middle_kernel import MiddleKernel
# from .late_kernel import LateKernel
# from .enhanced_middle_kernel import EnhancedMiddleKernel
from sklearn.preprocessing import LabelEncoder

class KernelConstructor:
    def __init__(self, kernel_level, method): #, kernel_method
        self.kernel_level = kernel_level
        if self.kernel_level == "early":
            self.kernel_method = EarlyKernel(kernel_type=method)
        # elif self.kernel_level == "multi-kernel":
        #     self.kernel_method = MultiKernel()
        # elif self.kernel_level == "middle":
        #     self.kernel_method = MiddleKernel(kernel_type=method)
        # elif self.kernel_level == "late":
        #     self.kernel_method = LateKernel(method=method)
        # elif self.kernel_level == "enhancedMiddle":
        #     self.kernel_method = EnhancedMiddleKernel(method=method)
        else:
            raise ValueError("Invalid kernel level")
        # self.kernel_method = kernel_method

        self.model = None
    def fit(self, X_train, y_train):
        return self.kernel_method.fit(X_train,y_train)

    def fit_transform(self, X, y):
        self.model = self.kernel_method.fit(X,y)
        K_X = self.model.transform(X)
        return K_X

    def transform(self, X, y=None):
        return self.model.transform(X)

    def create_ouput_kernel(self, y):
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(y)
        encoded_new_label = encoded_labels * 2 - 1
        y = encoded_new_label.reshape(1, -1)
        K = y.T @ y
        return  K

    def __str__(self):
        return self.kernel_method.__str__()
