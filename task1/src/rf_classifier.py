from interface import MNISTClassifierInterface
from sklearn.ensemble import RandomForestClassifier as RandomForest


class RandomForestMNISTClassifier(MNISTClassifierInterface):
    """
    Random Forest classifier implementation for MNIST.

    This classifier uses sklearn's RandomForestClassifier with optimized
    hyperparameters for digit classification.
    """

    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Initialize the Random Forest classifier.

        Args:
            n_estimators: Number of trees in the forest (default: 100)
            max_depth: Maximum depth of trees (default: 20)
            random_state: Random seed for reproducibility (default: 42)
        """
        self.model = RandomForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
            verbose=1
        )

    def train(self, X_train, y_train, epochs=None, batch_size=None):
        """
        Train the Random Forest model.

        Note: epochs and batch_size parameters are ignored for Random Forest
        but included for interface consistency.

        Args:
            X_train: Training images, shape (n_samples, height, width)
            y_train: Training labels, shape (n_samples)
            epochs: Not used for Random Forest
            batch_size: Not used for Random Forest
        """
        if len(X_train.shape) == 3:     # Flatten images if they are 2D (28x28)
            X_train = X_train.reshape(X_train.shape[0], -1)
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Make predictions on test data.

        Args:
            X_test: Test images, shape (n_samples, height, width)

        Returns:
            numpy.ndarray: Predicted labels, shape (n_samples)
        """
        if len(X_test.shape) == 3:      # Flatten images if they are 2D (28x28)
            X_test = X_test.reshape(X_test.shape[0], -1)
        return self.model.predict(X_test)