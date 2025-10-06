# Task 1: MNIST Image Classification with OOP

## Description

This part of the project implements three different MNIST digit classification models using Object-Oriented Programming principles. All models follow a common interface, which ensures consistent input/output behavior and allows switching between algorithms through a unified wrapper class.
## Implemented Models

1. **Random Forest (rf)** - Traditional machine learning approach using ensemble of decision trees
2. **Feed-Forward Neural Network (nn)** - Multi-layer perceptron with fully connected layers (PyTorch)
3. **Convolutional Neural Network (cnn)** - Deep learning model optimized for image data (PyTorch)

## Project Structure

```
task1/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/                               # Source code
│   ├── interface.py                  # Abstract base class
│   ├── rf_classifier.py   # Random Forest implementation
│   ├── ff_classifier.py  # Feed-Forward NN implementation
│   ├── cnn_classifier.py             # CNN implementation
│   └── mnist_classifier.py           # Unified wrapper class
├── notebooks/                         # Jupyter notebooks
│   └── demo.ipynb                    # Demonstration and comparison
├── data/                               # MNIST dataset
│   └── ... 
└── models/                            # Directory for saved models (optional)
```

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd task1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Architecture Details

### Random Forest
- **Algorithm**: Ensemble of 100 decision trees
- **Max Depth**: 20
- **Input**: Flattened 784-dimensional vectors (28×28 images)

### Feed-Forward Neural Network
- **Architecture**:
  - Input: 784 neurons (flattened image)
  - Hidden Layer 1: 512 neurons + ReLU + Dropout(0.3)
  - Hidden Layer 2: 256 neurons + ReLU + Dropout(0.3)
  - Hidden Layer 3: 128 neurons + ReLU + Dropout(0.2)
  - Output: 10 neurons (softmax)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Cross-Entropy

### Convolutional Neural Network
- **Architecture**:
  - Conv Layer 1: 32 filters (3×3) + ReLU + MaxPool(2×2)
  - Conv Layer 2: 64 filters (3×3) + ReLU + MaxPool(2×2)
  - Conv Layer 3: 64 filters (3×3) + ReLU
  - Flatten
  - Dense Layer: 128 neurons + ReLU + Dropout(0.5)
  - Output: 10 neurons (softmax)
- **Optimizer**: Adam (lr=0.001)
- **Loss**: Cross-Entropy

## Demo Notebook

For a comprehensive demonstration including:
- Data loading and preprocessing
- Training all three models
- Performance comparison
- Confusion matrices
- Edge case handling

Please refer to `notebooks/demo.ipynb`

## Edge Cases Handled

1. **Invalid algorithm name**: Raises `ValueError` with helpful message
2. **Different input shapes**: Automatic reshaping for compatibility
3. **Empty datasets**: Proper error handling
4. **GPU/CPU compatibility**: Automatic device selection for PyTorch models