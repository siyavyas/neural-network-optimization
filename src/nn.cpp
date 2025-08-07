#include <Eigen/Dense>
#include <cmath>
#include <fstream>
#include <iostream>
#include <optimization/activation.hpp>
#include <optimization/nn.hpp>
#include <optimization/optimization.hpp>
#include <random>


namespace neural_network {

NeuralNetwork::NeuralNetwork() {}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layers, activation::ActivationType activationType, double learningRate)
    : activationType_(activationType), learningRate(learningRate) {
      for (size_t i = 1; i < layers.size(); ++i) {
        weights.push_back(Eigen::MatrixXd::Random(layers[i], layers[i - 1]));
        biases.push_back(Eigen::MatrixXd::Random(layers[i], 1));
      }

      // resize
      d_weights.resize(weights.size());
      d_biases.resize(biases.size());
}

NeuralNetwork::~NeuralNetwork() {}

Eigen::MatrixXd NeuralNetwork::activate(const Eigen::MatrixXd& x) {
    if (activationType_ == activation::ActivationType::Sigmoid) {
        return activation::activationFunction::sigmoid(x);
    } else {
        return activation::activationFunction::relu(x);
    }
}

Eigen::MatrixXd NeuralNetwork::activateDeriv(const Eigen::MatrixXd& x) {
    if (activationType_ == activation::ActivationType::Sigmoid) {
        return activation::activationFunction::sigmoidDeriv(x);
    } else {
        return activation::activationFunction::reluDeriv(x);
    }
}

double NeuralNetwork::computeLoss(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    return (predicted - target).array().square().mean();
}

Eigen::MatrixXd NeuralNetwork::computeLossDeriv(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    return 2 * (predicted - target) / predicted.rows();
}

Eigen::MatrixXd NeuralNetwork::forward(const Eigen::MatrixXd& input) {
  layerInputs.clear();
  layerOutputs.clear();

  Eigen::MatrixXd output = input;

  // iterates through each layer
  for (size_t i = 0; i < weights.size(); ++i) {
    layerInputs.push_back(output);  // stores input of curr layer
    output = weights[i] * output + biases[i].replicate(1, input.cols());  // weighted sum of curr layer
    output = activate(output);
    layerOutputs.push_back(output);
  } 

  return output;
}

void NeuralNetwork::backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target) {

  Eigen::MatrixXd error = computeLossDeriv(layerOutputs.back(), target);

  for (int i = weights.size() - 1; i >= 0; --i) {
    Eigen::MatrixXd gradient = error.cwiseProduct(activateDeriv(layerOutputs[i]));
        
    d_weights[i] = gradient * layerInputs[i].transpose();
    d_biases[i] = gradient.rowwise().sum();

    if (i > 0) {
      error = (weights[i].transpose() * gradient).cwiseProduct(activateDeriv(layerInputs[i]));
    }
  }

}

void NeuralNetwork::updateWeights() {
  for (size_t i = 0; i < weights.size(); ++i) {
    weights[i] -= learningRate * d_weights[i];
    biases[i] -= learningRate * d_biases[i];
  }
}

void NeuralNetwork::train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, int epochs) {

  std::cout << "Training..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // For small datasets, train on the entire dataset
        Eigen::MatrixXd batchInput = input.transpose();
        Eigen::MatrixXd batchTarget = target.transpose();

        Eigen::MatrixXd output = forward(batchInput);
        backward(batchInput, batchTarget);
        updateWeights();

        // Compute loss for monitoring
        Eigen::MatrixXd fullOutput = forward(input.transpose());
        double epochLoss = computeLoss(fullOutput, target.transpose());

        if (epoch % 10 == 0 || epoch == epochs - 1) {
            std::cout << "Epoch " << epoch << " Loss: " << epochLoss << std::endl;
        }
    }
}

Eigen::MatrixXd NeuralNetwork::predict(const Eigen::MatrixXd& input) {
    return forward(input.transpose());
}

} // namespace neural_network