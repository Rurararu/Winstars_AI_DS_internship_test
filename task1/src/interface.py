from abc import ABC, abstractmethod


class MNISTClassifierInterface(ABC):
    """
    Abstract base class for MNIST classifiers.

    All concrete classifier implementations must inherit from this class
    and implement the train and predict methods.
    """
    @abstractmethod
    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the classifier on the provided training data.

        Args:
            X_train: Training features (images)
            y_train: Training labels
            epochs: Number of training epochs (default: 10)
            batch_size: Batch size for training (default: 32)
        """
        pass

    @abstractmethod
    def predict(self, X_test):
        """
        Make predictions on the test data.

        Args:
            X_test: Test features (images)

        Returns:
            numpy.ndarray: Predicted labels
        """
        pass