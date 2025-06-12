from sklearn.model_selection import KFold, StratifiedKFold, RepeatedKFold
import numpy as np


class CrossValidator:
    def __init__(self, n_splits=5, n_repeats=1, stratified=False, random_state=None):
        """
        Initialize the cross-validator.

        Args:
        - n_splits (int): Number of folds.
        - n_repeats (int): Number of repetitions for repeated k-fold. Default is 1.
        - stratified (bool): Whether to use stratified k-fold. Default is False.
        - random_state (int): Random state for reproducibility. Default is None.
        """
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.stratified = stratified
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

    def get_splits(self, X, y=None):
        """
        Get the train/test splits for the data.

        Args:
        - X (array-like): Feature matrix.
        - y (array-like, optional): Target values. Required for stratified k-fold.

        Returns:
        - generator: Train/test indices for each fold.
        """
        if self.n_repeats > 1:
            if self.stratified:
                # Custom Repeated Stratified K-Fold
                return self._repeated_stratified_kfold(X, y)
            else:
                return RepeatedKFold(
                    n_splits=self.n_splits,
                    n_repeats=self.n_repeats,
                    random_state=self.random_state
                ).split(X, y)
        elif self.stratified:
            if y is None:
                raise ValueError("y is required for stratified k-fold.")
            return StratifiedKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            ).split(X, y)
        else:
            return KFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=self.random_state
            ).split(X, y)

    def _repeated_stratified_kfold(self, X, y):
        """
        Custom implementation for Repeated Stratified K-Fold.

        Args:
        - X (array-like): Feature matrix.
        - y (array-like): Target values.

        Yields:
        - train/test indices for each split.
        """
        if y is None:
            raise ValueError("y is required for stratified k-fold.")

        for repeat in range(self.n_repeats):
            # Shuffle data indices for each repeat
            indices = np.arange(len(y))
            self.rng.shuffle(indices)
            X_shuffled = np.array(X)[indices]
            y_shuffled = np.array(y)[indices]

            # Use StratifiedKFold for the current repetition
            skf = StratifiedKFold(n_splits=self.n_splits,
                                  shuffle=True, random_state=self.random_state)
            for train_idx, test_idx in skf.split(X_shuffled, y_shuffled):
                # Map shuffled indices back to the original
                yield indices[train_idx], indices[test_idx]


# Example usage
if __name__ == "__main__":
    # Sample data
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 2, 100)  # Binary target

    # Initialize cross-validator
    cv = CrossValidator(n_splits=5, n_repeats=2,
                        stratified=True, random_state=42)

    # Generate splits
    for fold, (train_idx, test_idx) in enumerate(cv.get_splits(X, y)):
        print(f"Repeat {fold // 5 + 1}, Fold {fold % 5 + 1}")
        print(f"Train indices: {train_idx}")
        print(f"Test indices: {test_idx}")
        print("lenght of train indices: ", len(train_idx))
        print("lenght of test indices: ", len(test_idx))
        print("-" * 40)
