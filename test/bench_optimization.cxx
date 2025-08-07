#include <benchmark/benchmark.h>

#include <optimization/optimization.hpp>
#include <optimization/nn.hpp>
#include <optimization/activation.hpp>
#include <optimization/kFold.hpp>

#include <Eigen/Dense>
#include <random>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Benchmark: Neural Network Training with Sigmoid Activation
static void BM_NeuralNetworkTrainingSigmoid(benchmark::State& state) {
    neural_network::NeuralNetwork nn({4, 10, 1}, activation::ActivationType::Sigmoid, 0.01);

    // Create training data
    Matrix input(4, 4);
    input << 0, 1, 2, 3,
             1, 2, 3, 4,
             2, 3, 4, 5,
             3, 4, 5, 6;

    Matrix target(4, 1);
    target << 0, 1, 1, 0;

    for (auto _ : state) {
        nn.train(input, target, 100);
    }
}

// Benchmark: Neural Network Training with ReLU Activation
static void BM_NeuralNetworkTrainingRelu(benchmark::State& state) {
    neural_network::NeuralNetwork nn({4, 10, 1}, activation::ActivationType::Relu, 0.01);

    // Create training data
    Matrix input(4, 4);
    input << 0, 1, 2, 3,
             1, 2, 3, 4,
             2, 3, 4, 5,
             3, 4, 5, 6;

    Matrix target(4, 1);
    target << 0, 1, 1, 0;

    for (auto _ : state) {
        nn.train(input, target, 100);
    }
}

// Benchmark: Forward Pass Performance
static void BM_NeuralNetworkForwardPass(benchmark::State& state) {
    neural_network::NeuralNetwork nn({4, 10, 3}, activation::ActivationType::Sigmoid, 0.01);

    Matrix input(4, 1);
    input << 0.5, 0.3, 0.7, 0.2;

    for (auto _ : state) {
        Matrix output = nn.forward(input);
        benchmark::DoNotOptimize(output);
    }
}

// Benchmark: Activation Functions
static void BM_SigmoidActivation(benchmark::State& state) {
    Matrix input = Matrix::Random(100, 100);

    for (auto _ : state) {
        Matrix output = activation::activationFunction::sigmoid(input);
        benchmark::DoNotOptimize(output);
    }
}

static void BM_ReluActivation(benchmark::State& state) {
    Matrix input = Matrix::Random(100, 100);

    for (auto _ : state) {
        Matrix output = activation::activationFunction::relu(input);
        benchmark::DoNotOptimize(output);
    }
}

// Benchmark: K-Fold Cross Validation
static void BM_KFoldCrossValidation(benchmark::State& state) {
    // Create sample data
    Matrix data(100, 4);
    Matrix labels(100, 1);
    
    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 1.0);
    
    for (int i = 0; i < data.rows(); ++i) {
        for (int j = 0; j < data.cols(); ++j) {
            data(i, j) = dis(gen);
        }
        labels(i, 0) = static_cast<int>(dis(gen) * 3); // 0, 1, or 2
    }

    int k = state.range(0);

    for (auto _ : state) {
        auto folds = cv::kFold::split(data, labels, k);
        benchmark::DoNotOptimize(folds);
    }
}

// Benchmark: Neural Network Prediction
static void BM_NeuralNetworkPrediction(benchmark::State& state) {
    neural_network::NeuralNetwork nn({4, 10, 3}, activation::ActivationType::Sigmoid, 0.01);

    Matrix input(1, 4);
    input << 5.1, 3.5, 1.4, 0.2;

    for (auto _ : state) {
        Matrix prediction = nn.predict(input);
        benchmark::DoNotOptimize(prediction);
    }
}

// Benchmark: Large Neural Network Training
static void BM_LargeNeuralNetworkTraining(benchmark::State& state) {
    int inputSize = state.range(0);
    int hiddenSize = state.range(1);
    int outputSize = state.range(2);
    
    neural_network::NeuralNetwork nn({inputSize, hiddenSize, outputSize}, 
                                   activation::ActivationType::Sigmoid, 0.01);

    // Create training data
    Matrix input = Matrix::Random(10, inputSize);
    Matrix target = Matrix::Random(10, outputSize);

    for (auto _ : state) {
        nn.train(input, target, 50);
    }
}

// Register benchmarks
BENCHMARK(BM_NeuralNetworkTrainingSigmoid);
BENCHMARK(BM_NeuralNetworkTrainingRelu);
BENCHMARK(BM_NeuralNetworkForwardPass);
BENCHMARK(BM_SigmoidActivation);
BENCHMARK(BM_ReluActivation);
BENCHMARK(BM_NeuralNetworkPrediction);

// K-fold benchmarks with different k values
BENCHMARK(BM_KFoldCrossValidation)->Arg(5)->Arg(10);

// Large network benchmarks with different sizes
BENCHMARK(BM_LargeNeuralNetworkTraining)
    ->Args({10, 20, 5})
    ->Args({20, 40, 10})
    ->Args({50, 100, 20});

BENCHMARK_MAIN();
