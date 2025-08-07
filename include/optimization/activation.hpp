#ifndef OPTIMIZATION_ACTIVATION_HPP
#define OPTIMIZATION_ACTIVATION_HPP


#include <Eigen/Dense>

namespace activation { 

    enum class ActivationType {
        Sigmoid,
        Relu
    };

    class activationFunction {
    public:

        static Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& X); 
        static Eigen::MatrixXd sigmoidDeriv(const Eigen::MatrixXd& X);
        static Eigen::MatrixXd relu(const Eigen::MatrixXd& X);
        static Eigen::MatrixXd reluDeriv(const Eigen::MatrixXd& X);
        static Eigen::MatrixXd softmax(const Eigen::MatrixXd& X);

    };

} //namespace activation


#endif  // OPTIMIZATION_ACTIVATION_HPP