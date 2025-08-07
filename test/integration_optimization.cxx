#include <gtest/gtest.h>
#include <optimization/optimization.hpp>
#include <optimization/nn.hpp>
#include <optimization/activation.hpp>
#include <optimization/kFold.hpp>

#include <string>
#include <sstream>
#include <iostream>

#include <data/iris.h>

namespace {

    std::string GetIrisCSV() {
        return std::string(reinterpret_cast<const char*>(data_iris_csv), data_iris_csv_len);
    } 

} // namespace 

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Integration Test: Neural Network Training and Prediction
TEST(Integration, NeuralNetworkTrainingAndPrediction) {
    // Create a simple neural network for iris classification
    neural_network::NeuralNetwork nn({4, 10, 3}, activation::ActivationType::Sigmoid, 0.01);
    
    // Create training data (simplified iris features)
    // Each row is a sample, each column is a feature
    Matrix input(4, 4);
    input << 5.1, 3.5, 1.4, 0.2,  // Iris setosa
             7.0, 3.2, 4.7, 1.4,  // Iris versicolor  
             6.3, 3.3, 6.0, 2.5,  // Iris virginica
             4.9, 3.0, 1.4, 0.2;  // Iris setosa
    
    // Target matrix: each row is a sample, each column is a class
    Matrix target(4, 3);
    target << 1, 0, 0,  // Setosa: [1,0,0]
             0, 1, 0,  // Versicolor: [0,1,0]
             0, 0, 1,  // Virginica: [0,0,1]
             1, 0, 0;  // Setosa: [1,0,0]
    
    // Train the network
    EXPECT_NO_THROW(nn.train(input, target, 100));
    
    // Test prediction
    Matrix testInput(1, 4);
    testInput << 5.1, 3.5, 1.4, 0.2; // Iris setosa features
    
    Matrix prediction = nn.predict(testInput);
    
    // The neural network returns output in the format (output_features, samples)
    // So for 3 classes and 1 sample, we get 3x1
    EXPECT_EQ(prediction.rows(), 3);
    EXPECT_EQ(prediction.cols(), 1);
    
    // Check that prediction values are reasonable (between 0 and 1 for sigmoid)
    for (int i = 0; i < prediction.rows(); ++i) {
        EXPECT_GE(prediction(i, 0), 0.0);
        EXPECT_LE(prediction(i, 0), 1.0);
    }
}

// Integration Test: Activation Functions
TEST(Integration, ActivationFunctions) {
    Matrix input(2, 3);
    input << 1.0, -1.0, 0.5,
             0.0, -0.5, 2.0;
    
    // Test sigmoid activation
    Matrix sigmoidOutput = activation::activationFunction::sigmoid(input);
    EXPECT_EQ(sigmoidOutput.rows(), 2);
    EXPECT_EQ(sigmoidOutput.cols(), 3);
    
    // All sigmoid outputs should be between 0 and 1
    for (int i = 0; i < sigmoidOutput.rows(); ++i) {
        for (int j = 0; j < sigmoidOutput.cols(); ++j) {
            EXPECT_GE(sigmoidOutput(i, j), 0.0);
            EXPECT_LE(sigmoidOutput(i, j), 1.0);
        }
    }
    
    // Test ReLU activation
    Matrix reluOutput = activation::activationFunction::relu(input);
    EXPECT_EQ(reluOutput.rows(), 2);
    EXPECT_EQ(reluOutput.cols(), 3);
    
    // All ReLU outputs should be >= 0
    for (int i = 0; i < reluOutput.rows(); ++i) {
        for (int j = 0; j < reluOutput.cols(); ++j) {
            EXPECT_GE(reluOutput(i, j), 0.0);
        }
    }
    
    // Check specific ReLU behavior
    EXPECT_EQ(reluOutput(0, 0), 1.0);  // positive input
    EXPECT_EQ(reluOutput(0, 1), 0.0);  // negative input
    EXPECT_EQ(reluOutput(0, 2), 0.5);  // positive input
}

// Integration Test: Forward and Backward Pass
TEST(Integration, ForwardBackwardPass) {
    neural_network::NeuralNetwork nn({2, 3, 1}, activation::ActivationType::Sigmoid, 0.01);
    
    Matrix input(2, 1);  // 2 features, 1 sample
    input << 0.5, 0.3;
    
    Matrix target(1, 1);  // 1 output, 1 sample
    target << 1.0;
    
    // Test forward pass
    Matrix forwardOutput = nn.forward(input);
    EXPECT_EQ(forwardOutput.rows(), 1);
    EXPECT_EQ(forwardOutput.cols(), 1);
    EXPECT_GE(forwardOutput(0, 0), 0.0);
    EXPECT_LE(forwardOutput(0, 0), 1.0);
    
    // Test backward pass (should not crash)
    EXPECT_NO_THROW(nn.backward(input, target));
}

// Integration Test: Iris Data Loading
TEST(Integration, IrisDataLoading) {
    std::string csvData = GetIrisCSV();
    EXPECT_FALSE(csvData.empty());
    
    // Check that the CSV contains iris data
    EXPECT_TRUE(csvData.find("Iris-setosa") != std::string::npos);
    EXPECT_TRUE(csvData.find("Iris-versicolor") != std::string::npos);
    EXPECT_TRUE(csvData.find("Iris-virginica") != std::string::npos);
    
    // Check that it has the expected structure
    std::istringstream stream(csvData);
    std::string firstLine;
    std::getline(stream, firstLine);
    
    // Should have comma-separated values
    EXPECT_TRUE(firstLine.find(',') != std::string::npos);
}

// Integration Test: K-Fold Cross Validation
TEST(Integration, KFoldCrossValidation) {
    // Create sample data
    Matrix data(10, 2);
    data << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
            11, 12, 13, 14, 15, 16, 17, 18, 19, 20;
    
    Matrix labels(10, 1);
    labels << 0, 1, 0, 1, 0, 1, 0, 1, 0, 1;
    
    // Test k-fold splitting
    auto folds = cv::kFold::split(data, labels, 5);
    
    EXPECT_EQ(folds.size(), 5);
    
    // Each fold should have 2 samples (10 samples / 5 folds)
    for (const auto& fold : folds) {
        EXPECT_EQ(fold.first.rows(), 2);
        EXPECT_EQ(fold.second.rows(), 2);
        EXPECT_EQ(fold.first.cols(), 2);
        EXPECT_EQ(fold.second.cols(), 1);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
