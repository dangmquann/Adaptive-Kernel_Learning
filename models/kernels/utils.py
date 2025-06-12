from .kernel_combination import KernelCombination

def train_test_DL(X_train, Y_train, X_test, Y_test, method, num_kernels, epochs=100, batch_size=32, lr=0.001, patience=10, pretrain_epochs=50, lambda_reg=0.1,
                  use_fe=True):
    model = KernelCombination(
        method=method,
        num_kernels=num_kernels,
        lr=lr,
        epochs=epochs,
        pretrain_epochs=pretrain_epochs,
        batch_size=batch_size,
        patience=patience,
        lambda_reg=lambda_reg,
        use_fe=use_fe
    )

    model.fit(X_train, Y_train, X_test, Y_test)

    kernels_train = model.transform(X_train)
    kernels_train_test = model.transform(X_test)
    kernels_test = model.transform(X_test, compute_self_kernel=True)

    return kernels_train, kernels_train_test, kernels_test

