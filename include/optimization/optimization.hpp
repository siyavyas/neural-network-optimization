#ifndef OPTIMIZATION_OPTIMIZATION_HPP
#define OPTIMIZATION_OPTIMIZATION_HPP

#include <Eigen/Dense>

namespace optimizer {

    class SGD {
    public:

        SGD(double learningRate);
        void update(Eigen::MatrixXd& weights, const Eigen::MatrixXd& gradients);
        void update(Eigen::VectorXd& weights, const Eigen::VectorXd& gradients);

    private:
        double learningRate;

    };

} // namespace optimizer

#endif // OPTIMIZATION_OPTIMIZATION_HPP