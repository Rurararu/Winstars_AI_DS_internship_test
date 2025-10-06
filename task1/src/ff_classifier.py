import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from interface import MNISTClassifierInterface

class FeedForwardNN(nn.Module):
    """
        PyTorch Feed-Forward Neural Network architecture:
            - Input: 784 features (28x28)
            - Hidden Layer 1: 512 neurons + ReLU + Dropout(0.3)
            - Hidden Layer 2: 256 neurons + ReLU + Dropout(0.3)
            - Hidden Layer 3: 128 neurons + ReLU + Dropout(0.3)
            - Output: 10 neurons (digits 0-9)
        """
    def __init__(self):
        super(FeedForwardNN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.2)

    def forward(self, x):
        x = self.flatten(x)     # Flatten input
        x = self.relu1(self.fc1(x))
        x = self.dropout1(x)

        x = self.relu2(self.fc2(x))
        x = self.dropout2(x)

        x = self.relu3(self.fc3(x))
        x = self.dropout3(x)

        x = self.fc4(x)
        return x

class FeedForwardMNISTClassifier(MNISTClassifierInterface):
    """
    Feed-Forward Neural Network classifier for MNIST using PyTorch
    """
    def __init__(self, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FeedForwardNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        print(f"Device: {self.device}")

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the Feed-Forward Neural Network

        Args:
            X_train: Training images, shape (n_samples, height, width) or (n_samples, features)
            y_train: Training labels, shape (n_samples,)
            epochs: Number of training epochs (default: 10)
            batch_size: Batch size for training (default: 32)
        """
        if len(X_train.shape) == 3:     # Flatten images if they are 2D
            X_train = X_train.reshape(X_train.shape[0], -1)

        X_tensor = torch.FloatTensor(X_train).to(self.device)
        y_tensor = torch.LongTensor(y_train).to(self.device)

        dataloader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=batch_size, shuffle=True)

        self.model.train()
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0

            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

                avg_loss = total_loss / len(dataloader)
                accuracy = 100 * correct / total
            print(f"Epoch: {epoch+1}/{epochs}, Loss: {avg_loss}, Accuracy: {accuracy}")


    def predict(self, X_test):
        """
        Make predictions on test data

        Args:
            X_test: Test images, shape (n_samples, height, width) or (n_samples, features)

        Returns:
            numpy.ndarray: Predicted labels, shape (n_samples)
        """
        if len(X_test.shape) == 3:      # Flatten images if they are 2D
            X_test = X_test.reshape(X_test.shape[0], -1)

        X_tensor = torch.FloatTensor(X_test).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted