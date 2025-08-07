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

// Basic Neural Network Tests
TEST(NeuralNetwork, Constructor) {
    std::vector<int> layers = {4, 10, 3};
    neural_network::NeuralNetwork nn(layers, activation::ActivationType::Sigmoid, 0.01);
    
    EXPECT_EQ(nn.weights.size(), 2); // 2 weight matrices for 3 layers
    EXPECT_EQ(nn.biases.size(), 2);  // 2 bias vectors for 3 layers
    EXPECT_EQ(nn.activationType_, activation::ActivationType::Sigmoid);
    EXPECT_EQ(nn.learningRate, 0.01);
}

TEST(NeuralNetwork, SigmoidActivation) {
    neural_network::NeuralNetwork nn({4, 10, 1}, activation::ActivationType::Sigmoid, 0.01);

    Matrix input(1, 4);
    input << 0, 1, 2, 3;

    Matrix output = nn.predict(input);

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 1);
    EXPECT_GE(output(0, 0), 0.0); // Sigmoid output should be between 0 and 1
    EXPECT_LE(output(0, 0), 1.0);
}

TEST(NeuralNetwork, ReluActivation) {
    neural_network::NeuralNetwork nn({4, 10, 1}, activation::ActivationType::Relu, 0.01);

    Matrix input(1, 4);
    input << 0, 1, 2, 3;

    Matrix output = nn.predict(input);

    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 1);
    EXPECT_GE(output(0, 0), 0.0); // ReLU output should be >= 0
}

TEST(NeuralNetwork, TrainingWithLoss) {
    neural_network::NeuralNetwork nn({4, 10, 1}, activation::ActivationType::Sigmoid, 0.01);

    Matrix input(1, 4);
    input << 0, 1, 2, 3;

    Matrix target(1, 1);
    target << 1;

    // Training should not crash
    EXPECT_NO_THROW(nn.train(input, target, 10));
}

// Activation Function Tests
TEST(Activation, Sigmoid) {
    Matrix input(2, 2);
    input << 1.0, -1.0, 0.5, -0.5;
    
    Matrix output = activation::activationFunction::sigmoid(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 2);
    
    // Sigmoid output should be between 0 and 1
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            EXPECT_GE(output(i, j), 0.0);
            EXPECT_LE(output(i, j), 1.0);
        }
    }
}

TEST(Activation, Relu) {
    Matrix input(2, 2);
    input << 1.0, -1.0, 0.5, -0.5;
    
    Matrix output = activation::activationFunction::relu(input);
    
    EXPECT_EQ(output.rows(), 2);
    EXPECT_EQ(output.cols(), 2);
    
    // ReLU output should be >= 0
    for (int i = 0; i < output.rows(); ++i) {
        for (int j = 0; j < output.cols(); ++j) {
            EXPECT_GE(output(i, j), 0.0);
        }
    }
    
    // Positive inputs should remain positive
    EXPECT_EQ(output(0, 0), 1.0);
    EXPECT_EQ(output(1, 0), 0.5);
    
    // Negative inputs should become 0
    EXPECT_EQ(output(0, 1), 0.0);
    EXPECT_EQ(output(1, 1), 0.0);
}

// Basic Forward Pass Test
TEST(NeuralNetwork, ForwardPass) {
    neural_network::NeuralNetwork nn({2, 3, 1}, activation::ActivationType::Sigmoid, 0.01);

    Matrix input(2, 1);  // 2 features, 1 sample
    input << 0.5, 0.3;

    Matrix output = nn.forward(input);
    
    EXPECT_EQ(output.rows(), 1);
    EXPECT_EQ(output.cols(), 1);
    EXPECT_GE(output(0, 0), 0.0);
    EXPECT_LE(output(0, 0), 1.0);
}

// Test Iris CSV Data Loading
TEST(IrisData, CSVDataAvailable) {
    std::string csvData = GetIrisCSV();
    EXPECT_FALSE(csvData.empty());
    EXPECT_GT(csvData.length(), 100); // Should have substantial data
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
