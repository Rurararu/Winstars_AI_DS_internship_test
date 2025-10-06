import joblib
import torch

from rf_classifier import RandomForestMNISTClassifier
from ff_classifier import FeedForwardMNISTClassifier
from cnn_classifier import ConvMNISTClassifier

class MnistClassifier:
    """
    Unified wrapper for MNIST classifiers.

    This class provides a consistent interface for different classification
    algorithms. Users can select the algorithm by passing the algorithm name
    to the constructor.

    Supported algorithms:
        - 'rf': Random Forest
        - 'nn': Feed-Forward Neural Network
        - 'cnn': Convolutional Neural Network
    """
    def __init__(self, algorithm="rf"):
        """
        Initialize the MNIST classifier with the specified algorithm.

        Args:
            algorithm: Classification algorithm to use ('rf', 'nn', or 'cnn')

        Raises:
            ValueError: If the specified algorithm is not supported
        """
        if algorithm == "rf":
            self.model = RandomForestMNISTClassifier()
        elif algorithm == "nn":
            self.model = FeedForwardMNISTClassifier()
        elif algorithm == "cnn":
            self.model = ConvMNISTClassifier()
        else:
            raise ValueError("Unknown algorithm. Use 'rf', 'nn', or 'cnn'.")

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the selected classifier

        Args:
            X_train: Training images
            y_train: Training labels
            epochs: Number of training epochs (ignored for Random Forest)
            batch_size: Batch size for training (ignored for Random Forest)
        """
        self.model.train(X_train, y_train, epochs, batch_size)


    def predict(self, X_test):
        """
        Make predictions using the selected classifier

        Args:
            X_test: Test images

        Returns:
            numpy.ndarray: Predicted labels
        """
        return self.model.predict(X_test)

    def save(self, path, algorithm="rf"):
        """
        Save the trained model to a file

        Args:
            path: Path to save the model (e.g., 'model_rf.pkl' or 'cnn_model.pth')

        Note:
            - For Random Forest: saves as .pkl file using joblib
            - For Neural Networks (nn, cnn): saves as .pth file using torch
        """
        if algorithm == "rf":
            joblib.dump(self.model, path)
        elif algorithm in ["nn", "cnn"]:
            torch.save(self.model.model.state_dict(), path)
        else:
            raise ValueError("Unsupported model type for saving.")