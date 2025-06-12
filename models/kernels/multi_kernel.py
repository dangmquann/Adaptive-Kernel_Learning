import numpy as np
from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
from models.EasyMKL import EasyMKL
from cvxopt import matrix

"""
@author: Michele Donini
@email: mdonini@math.unipd.it

EasyMKL: a scalable multiple kernel learning algorithm
by Fabio Aiolli and Michele Donini

Paper @ http://www.math.unipd.it/~mdonini/publications.html
"""

from cvxopt import matrix, solvers, mul
import numpy as np


class EasyMKL():
    ''' EasyMKL is a Multiple Kernel Learning algorithm.
        The parameter lam (lambda) has to be validated from 0 to 1.

        For more information:
        EasyMKL: a scalable multiple kernel learning algorithm
            by Fabio Aiolli and Michele Donini

        Paper @ http://www.math.unipd.it/~mdonini/publications.html
    '''
    def __init__(self, lam = 0.1, tracenorm = True):
        self.lam = lam
        self.tracenorm = tracenorm
        
        self.list_Ktr = None
        self.labels = None
        self.gamma = None
        self.weights = None
        self.traces = []

    def sum_kernels(self, list_K, weights = None):
        ''' Returns the kernel created by averaging of all the kernels '''
        k = matrix(0.0,(list_K[0].size[0],list_K[0].size[1]))
        if weights == None:
            for ker in list_K:
                k += ker
        else:
            for w,ker in zip(weights,list_K):
                k += w * ker            
        return k
    
    def traceN(self, k):
        return sum([k[i,i] for i in range(k.size[0])]) / k.size[0]
    
    def train(self, list_Ktr, labels):
        ''' 
            list_Ktr : list of kernels of the training examples
            labels : array of the labels of the training examples
        '''
        self.traces = []
        self.weights = None
        self.list_Ktr = list_Ktr  
        for k in self.list_Ktr:
            self.traces.append(self.traceN(k))
        if self.tracenorm:
            self.list_Ktr = [k / self.traceN(k) for k in list_Ktr]

        set_labels = set(labels)
        if len(set_labels) != 2:
            print ('The different labels are not 2')
            return None
        elif (-1 in set_labels and 1 in set_labels):
            self.labels = labels
        else:
            poslab = max(set_labels)
            self.labels = matrix(np.array([1. if i==poslab else -1. for i in labels]))
        
        # Sum of the kernels
        ker_matrix = matrix(self.sum_kernels(self.list_Ktr))

        YY = matrix(np.diag(list(matrix(self.labels))))
        KLL = (1.0-self.lam)*YY*ker_matrix*YY
        LID = matrix(np.diag([self.lam]*len(self.labels)))
        Q = 2*(KLL+LID)
        p = matrix([0.0]*len(self.labels))
        G = -matrix(np.diag([1.0]*len(self.labels)))
        h = matrix([0.0]*len(self.labels),(len(self.labels),1))
        A = matrix([[1.0 if lab==+1 else 0 for lab in self.labels],[1.0 if lab2==-1 else 0 for lab2 in self.labels]]).T
        b = matrix([[1.0],[1.0]],(2,1))
        
        solvers.options['show_progress']=False#True
        sol = solvers.qp(Q,p,G,h,A,b)
        # Gamma:
        self.gamma = sol['x']     
        
        # Bias for classification:
        bias = 0.5 * self.gamma.T * ker_matrix * YY * self.gamma
        self.bias = bias

        # Weights evaluation:
        yg =  mul(self.gamma.T,self.labels.T)
        self.weights = []
        for kermat in self.list_Ktr:
            b = yg*kermat*yg.T
            self.weights.append(b[0])
            
        norm2 = sum([w for w in self.weights])
        self.weights = [w / norm2 for w in self.weights]

        if self.tracenorm: 
            for idx,val in enumerate(self.traces):
                self.weights[idx] = self.weights[idx] / val        
        
        if True:
            ker_matrix = matrix(self.sum_kernels(self.list_Ktr, self.weights))
            YY = matrix(np.diag(list(matrix(self.labels))))
            
            KLL = (1.0-self.lam)*YY*ker_matrix*YY
            LID = matrix(np.diag([self.lam]*len(self.labels)))
            Q = 2*(KLL+LID)
            p = matrix([0.0]*len(self.labels))
            G = -matrix(np.diag([1.0]*len(self.labels)))
            h = matrix([0.0]*len(self.labels),(len(self.labels),1))
            A = matrix([[1.0 if lab==+1 else 0 for lab in self.labels],[1.0 if lab2==-1 else 0 for lab2 in self.labels]]).T
            b = matrix([[1.0],[1.0]],(2,1))
            
            solvers.options['show_progress']=False#True
            sol = solvers.qp(Q,p,G,h,A,b)
            # Gamma:
            self.gamma = sol['x']
        
        
        return self
    
    def rank(self,list_Ktest):
        '''
            list_Ktr : list of kernels of the training examples
            labels : array of the labels of the training examples
            Returns the list of the examples in test set of the kernel K ranked
        '''
        if self.weights == None:
            print ('EasyMKL has to be trained first!')
            return
         
        #YY = matrix(np.diag(self.labels).copy())
        YY = matrix(np.diag(list(matrix(self.labels))))
        ker_matrix = matrix(self.sum_kernels(list_Ktest, self.weights))
        z = ker_matrix*YY*self.gamma
        return z


class MultiKernel:
    def __init__(self, gamma_rbf=1.0, gamma_poly=1.0, degree=3, coef0=1, lam=0.1, tracenorm=True):
        """
        Khởi tạo MultiKernel với các tham số cho kernel RBF và polynomial, cũng như các tham số cho EasyMKL.

        Args:
            gamma_rbf (float): Tham số gamma cho kernel RBF.
            gamma_poly (float): Tham số gamma cho kernel polynomial.
            degree (int): Bậc của kernel polynomial.
            coef0 (float): Hệ số coef0 của kernel polynomial.
            lam (float): Tham số lambda của EasyMKL (từ 0 đến 1).
            tracenorm (bool): Thực hiện tracenorm hay không trong EasyMKL.
        """
        self.gamma_rbf = gamma_rbf
        self.gamma_poly = gamma_poly
        self.degree = degree
        self.coef0 = coef0
        self.lam = lam
        self.tracenorm = tracenorm
        self.weights = None
        self.X_fit = None
        self.easyMKL = EasyMKL(lam=self.lam, tracenorm=self.tracenorm)

    def compute_kernel_list(self, X1, X2):
        """
        Tính toán danh sách các kernel (linear, RBF, polynomial) giữa hai ma trận đặc trưng.

        Args:
            X1 (np.ndarray): Ma trận đặc trưng thứ nhất.
            X2 (np.ndarray): Ma trận đặc trưng thứ hai.

        Returns:
            list: Danh sách các kernel dưới dạng cvxopt.matrix.
        """
        K_linear = linear_kernel(X1, X2)
        K_rbf = rbf_kernel(X1, X2, gamma=self.gamma_rbf)
        K_poly = polynomial_kernel(X1, X2, degree=self.degree, coef0=self.coef0, gamma=self.gamma_poly)
        K_rbf1 = rbf_kernel(X1, X2, gamma=0.5)
        K_poly1 = polynomial_kernel(X1, X2, degree=2, coef0=1, gamma=3.0)
        return [matrix(K_linear), matrix(K_rbf), matrix(K_poly), matrix(K_rbf1), matrix(K_poly1)]

    def fit(self, X, y=None):
        """
        Huấn luyện model bằng cách lưu dữ liệu huấn luyện và, nếu có, sử dụng EasyMKL để tính trọng số.

        Args:
            X (np.ndarray): Ma trận đặc trưng huấn luyện.
            y (np.ndarray, optional): Nhãn mục tiêu. Nếu có, trọng số sẽ được tối ưu thông qua EasyMKL.
        """
        self.X_fit = X
        if y is not None:
            # Tính danh sách kernel trên dữ liệu huấn luyện
            list_Ktr = self.compute_kernel_list(X, X)
            # Huấn luyện EasyMKL để tính trọng số
            self.easyMKL.train(list_Ktr, y)
            self.weights = {
                "linear": self.easyMKL.weights[0],
                "rbf": self.easyMKL.weights[1],
                "polynomial": self.easyMKL.weights[2],
                "rbf1": self.easyMKL.weights[3],
                "polynomial1": self.easyMKL.weights[4]
            }
        else:
            # Nếu không có nhãn thì dùng trọng số mặc định bằng nhau
            self.weights = {"linear": 1/4, "rbf": 1/4, "polynomial": 1/4, "rbf1": 1/4}
        return self

    def get_kernel(self, X1, X2):
        """
        Tính ma trận kernel kết hợp giữa X1 và X2 bằng cách kết hợp các kernel linear, RBF, polynomial với trọng số đã học.

        Args:
            X1 (np.ndarray): Ma trận đặc trưng thứ nhất.
            X2 (np.ndarray): Ma trận đặc trưng thứ hai.

        Returns:
            np.ndarray: Ma trận kernel kết hợp.
        """
        K_linear = linear_kernel(X1, X2)
        K_rbf = rbf_kernel(X1, X2, gamma=self.gamma_rbf)
        K_rbf1 = rbf_kernel(X1, X2, gamma=0.5)
        K_poly = polynomial_kernel(X1, X2, degree=self.degree, coef0=self.coef0, gamma=self.gamma)
        K_poly1 = polynomial_kernel(X1, X2, degree=2, coef0=1, gamma=3.0)
        # Nếu trọng số chưa được tính thì dùng mặc định
        if self.weights is None:
            print("Trọng số chưa được học. Sử dụng trọng số mặc định.")
            weights = {"linear": 1/5, "rbf": 1/5, "polynomial": 1/5, "rbf1": 1/5, "polynomial1": 1/5}
        else:
            weights = self.weights
        print(weights)
        K_combined = (weights["linear"] * K_linear +
                      weights["rbf"] * K_rbf +
                      weights["polynomial"] * K_poly +
                      weights["rbf1"] * K_rbf1 + 
                      weights["polynomial1"] * K_poly1)
        return K_combined

    def transform(self, X):
        """
        Biến đổi dữ liệu đầu vào sang không gian kernel kết hợp dựa trên dữ liệu huấn luyện.

        Args:
            X (np.ndarray): Ma trận đặc trưng cần biến đổi.

        Returns:
            tuple: (kernel_train, kernel_test)
                   - kernel_train: Ma trận kernel giữa dữ liệu huấn luyện và X.
                   - kernel_test: Ma trận kernel giữa X và chính nó.
        """
        if self.X_fit is None:
            raise ValueError("Model chưa được huấn luyện. Vui lòng gọi hàm fit trước.")
        kernel_train = self.get_kernel(self.X_fit, X)
        kernel_test = self.get_kernel(X, X)
        return kernel_train, kernel_test

    def fit_transform(self, X, y=None):
        """
        Huấn luyện model và biến đổi dữ liệu sang không gian kernel kết hợp.

        Args:
            X (np.ndarray): Ma trận đặc trưng.
            y (np.ndarray, optional): Nhãn mục tiêu.

        Returns:
            tuple: (kernel_train, kernel_test)
        """
        self.fit(X, y)
        return self.transform(X)

    def __str__(self):
        return (f"MultiKernel(weights={self.weights}, gamma_rbf={self.gamma_rbf}, "
                f"gamma_poly={self.gamma_poly}, degree={self.degree}, coef0={self.coef0}, "
                f"lam={self.lam}, tracenorm={self.tracenorm})")

# Ví dụ sử dụng:
if __name__ == '__main__':
    # Giả sử ta có dữ liệu huấn luyện X_train (numpy.ndarray) và nhãn y_train
    # Ví dụ tạo dữ liệu ngẫu nhiên:
    X_train = np.random.rand(50, 5)
    y_train = np.random.choice([-1, 1], size=50)
    
    # Khởi tạo và huấn luyện MultiKernel sử dụng EasyMKL để tính trọng số
    mk = MultiKernel(gamma_rbf=0.5, gamma_poly=0.5, degree=2, coef0=1, lam=0.1, tracenorm=True)
    mk.fit(X_train, y_train)
    
    # In ra trọng số đã học:
    print("Trọng số các kernel:", mk.weights)
    
    # Biến đổi dữ liệu (ví dụ với cùng X_train)
    K_train, K_test = mk.transform(X_train)
    print("Kích thước K_train:", K_train.shape)
    print("Kích thước K_test:", K_test.shape)
