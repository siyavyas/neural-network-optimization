#include <Eigen/Dense>
#include <optimization/kFold.hpp>

namespace cv {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

std::vector<std::pair<Matrix, Matrix>> kFold::split(const Matrix &data,
                                                    const Matrix &labels,
                                                    int k) {
  std::vector<std::pair<Matrix, Matrix>> folds;
  int foldSize = data.rows() / k;

  for (int i = 0; i < k; ++i) {
    Matrix foldData = data.block(i * foldSize, 0, foldSize, data.cols());
    Matrix foldLabels = labels.block(i * foldSize, 0, foldSize, 1);
    folds.push_back({foldData, foldLabels});
  }

  return folds;
}

}  // namespace cv
