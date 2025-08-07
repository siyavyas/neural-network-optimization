# Neural Network Optimization Project

A comprehensive C++ implementation of neural networks with optimization algorithms, featuring machine learning evaluation on the Iris dataset.

## Features

- **Neural Network Implementation**: Custom neural network with configurable layers and activation functions
- **Activation Functions**: Sigmoid, ReLU, and Softmax activation functions
- **Machine Learning Evaluation**: Complete evaluation pipeline with accuracy, precision, recall, F1-score, and confusion matrix
- **Cross-Validation**: K-fold cross-validation for robust model assessment
- **Stratified Sampling**: Balanced data splitting for fair evaluation
- **Performance Benchmarking**: Google Benchmark integration for performance analysis

## Project Structure

```
optimization/
├── include/
│   ├── models/           # Data management and ML model interfaces
│   └── optimization/     # Core neural network and activation functions
├── src/                  # Implementation files
├── test/                 # Unit tests, integration tests, and evaluation
│   └── data/            # Iris dataset
├── extern/               # External dependencies (Eigen, Google Test, Benchmark)
└── target/              # Build outputs
```

## Dependencies

- **Eigen3**: Linear algebra operations
- **Google Test**: Unit testing framework
- **Google Benchmark**: Performance benchmarking
- **CMake**: Build system

## Building the Project

### Prerequisites
- C++17 compatible compiler
- CMake 3.10 or higher
- Git

### Build Instructions

```bash
# Clone the repository
git clone <your-repo-url>
cd optimization

# Configure and build
cmake -B build
make -C build

# Run tests
./target/bin/test_optimization
./target/bin/integration_optimization
./target/bin/bench_optimization
./target/bin/evaluate_iris_classifier
```

## Evaluation Results

The neural network classifier achieves excellent performance on the Iris dataset:

### Test Set Performance
- **Overall Accuracy**: 96.67%
- **Setosa Classification**: 100% precision, 100% recall
- **Versicolor Classification**: 100% precision, 90% recall  
- **Virginica Classification**: 90.9% precision, 100% recall

### Cross-Validation Results
- **Mean Accuracy**: 76.00% ± 18.79%
- **Best Fold**: 100% accuracy
- **Worst Fold**: 53.33% accuracy

## Testing

### Unit Tests
```bash
./target/bin/test_optimization
```
Tests core neural network functionality including:
- Constructor validation
- Forward pass computation
- Activation functions
- Loss calculation

### Integration Tests
```bash
./target/bin/integration_optimization
```
Tests complete neural network workflows:
- Training and prediction
- Forward and backward propagation
- Matrix operations

### Performance Benchmarks
```bash
./target/bin/bench_optimization
```
Benchmarks neural network performance:
- Training speed
- Forward pass efficiency
- Activation function performance
- Cross-validation timing

### Iris Dataset Evaluation
```bash
./target/bin/evaluate_iris_classifier
```
Comprehensive evaluation including:
- Stratified data splitting
- Neural network training
- Classification metrics
- Confusion matrix
- K-fold cross-validation

## Neural Network Architecture

### Model Configuration
- **Input Layer**: 4 features (sepal length, sepal width, petal length, petal width)
- **Hidden Layer**: 10 neurons with sigmoid activation
- **Output Layer**: 3 neurons (one-hot encoded classes)
- **Learning Rate**: 0.01
- **Training Epochs**: 1000

### Activation Functions
- **Sigmoid**: σ(x) = 1 / (1 + e^(-x))
- **ReLU**: max(0, x)
- **Softmax**: e^(x_i) / Σ(e^(x_j))

## Metrics Calculation

### Classification Metrics
- **Accuracy**: (True Positives + True Negatives) / Total
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Cross-Validation
- **K-Fold**: 5-fold stratified cross-validation
- **Stratified Sampling**: Maintains class distribution across folds
- **Statistical Analysis**: Mean and standard deviation of accuracy

## Technical Details

### Data Processing
- **CSV Parsing**: Custom CSV parser for Iris dataset
- **Feature Scaling**: Raw feature values (no normalization applied)
- **One-Hot Encoding**: Target labels converted to one-hot vectors
- **Stratified Splitting**: 80% training, 20% testing with balanced classes

### Neural Network Implementation
- **Backpropagation**: Gradient descent optimization
- **Loss Function**: Mean squared error
- **Weight Updates**: Stochastic gradient descent
- **Matrix Operations**: Eigen library for efficient linear algebra

## Code Quality

### Design Principles
- **Modular Architecture**: Separated concerns (data, models, evaluation)
- **Type Safety**: Strong typing with Eigen matrices
- **Error Handling**: Robust file loading and validation
- **Documentation**: Comprehensive comments and clear function names

### Testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Benchmarking critical operations
- **Evaluation Tests**: Real-world dataset validation

## Future Enhancements

- [ ] Support for different activation functions
- [ ] Mini-batch gradient descent
- [ ] Regularization techniques (L1/L2)
- [ ] Dropout for overfitting prevention
- [ ] Support for different datasets
- [ ] GPU acceleration with CUDA
- [ ] Model serialization/deserialization
- [ ] Hyperparameter optimization

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: This project demonstrates a complete machine learning pipeline from data loading to model evaluation, showcasing both theoretical understanding and practical implementation skills in C++.
