#include <Eigen/Dense>
#include <optimization/optimization.hpp>
#include <random>

namespace optimizer {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

SGD::SGD(double learningRate) : learningRate(learningRate) {}

void optimizer::SGD::update(Matrix &weights, const Matrix &gradients) {
  /* for(int i = 0; i < weights.rows(); i++) {
      for(int j = 0; j < weights.cols(); j++) {
          weights(i, j) -= learningRate * grads(i, j);
      }
  } */

  weights -= learningRate * gradients;
}

void optimizer::SGD::update(Vector &weights, const Vector &gradients) {
  /* for(int i = 0; i < weights.size(); i++) {
      weights(i) -= learningRate * grads(i);
  } */

  weights -= learningRate * gradients;
}

}  // namespace optimizer
