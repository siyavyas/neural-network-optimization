#include <Eigen/Dense>
#include <optimization/activation.hpp>

using Matrix = Eigen::MatrixXd;

namespace activation {

  Matrix activation::activationFunction::sigmoid(const Matrix &X) {
    // element-wise
    return 1.0 / (1.0 + (-X.array()).exp());
  }

  Matrix activation::activationFunction::sigmoidDeriv(const Matrix &X) {
    Matrix sigmoid_X = activation::activationFunction::sigmoid(X);
    return sigmoid_X.array() * (1.0 - sigmoid_X.array());
  }

  Matrix activation::activationFunction::relu(const Matrix &X) {
    return X.array().max(0.0);
  }

  Matrix activation::activationFunction::reluDeriv(const Matrix &X) {
    Matrix relu_X = activation::activationFunction::relu(X);
    return (X.array() > 0.0).cast<double>();
  } 

  Matrix activation::activationFunction::softmax(const Matrix &X) {
    Matrix exp_X = X.array().exp();
    return exp_X.array().colwise() / exp_X.array().rowwise().sum();
  }

}  // namespace activation