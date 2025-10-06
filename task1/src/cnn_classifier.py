import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from interface import MNISTClassifierInterface

class ConvNN(nn.Module):
    """
    PyTorch Convolutional Neural Network architecture:
        - Conv Layer 1: 32 filters, 3x3 kernel + ReLU + MaxPool(2x2)
        - Conv Layer 2: 64 filters, 3x3 kernel + ReLU + MaxPool(2x2)
        - Conv Layer 3: 64 filters, 3x3 kernel + ReLU
        - Flatten
        - Dense Layer 1: 128 neurons + ReLU + Dropout(0.5)
        - Output: 10 neurons (digits 0-9)
    """
    def __init__(self):
        super(ConvNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(64 * 7 * 7, 128)   # After two pooling layers: 28x28 -> 14x14 -> 7x7
        self.fc2 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(1)  # (batch, 28, 28) -> (batch, 1, 28, 28)

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.conv3(x))

        x = x.view(-1, 64 * 7 * 7)  # Flatten

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class ConvMNISTClassifier(MNISTClassifierInterface):
    """
    Convolutional Neural Network classifier for MNIST using PyTorch.
    """
    def __init__(self, learning_rate=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ConvNN().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        print(f"Device: {self.device}")

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        """
        Train the CNN

        Args:
            X_train: Training images, shape (n_samples, height, width)
            y_train: Training labels, shape (n_samples,)
            epochs: Number of training epochs (default: 10)
            batch_size: Batch size for training (default: 32)
        """
        if len(X_train.shape) == 3:     # Ensure images are 3D
            X_train = X_train.reshape(-1, 28, 28)

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
            X_test: Test images, shape (n_samples, height, width)

        Returns:
            numpy.ndarray: Predicted labels, shape (n_samples)
        """
        if len(X_test.shape) == 3:      # Ensure images are 3D
            X_test = X_test.reshape(-1, 28, 28)

        X_tensor = torch.FloatTensor(X_test).to(self.device)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)

        return predicted