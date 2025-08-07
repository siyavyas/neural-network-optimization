#ifndef OPTIMIZATION_NN_HPP
#define OPTIMIZATION_NN_HPP

#include <Eigen/Dense>
#include <fstream>
#include <optimization/activation.hpp>


namespace neural_network {

class NeuralNetwork {
public:

    NeuralNetwork();
    ~NeuralNetwork();
    NeuralNetwork(const std::vector<int>& layers, activation::ActivationType activationType, double learningRate);
    Eigen::MatrixXd forward(const Eigen::MatrixXd& input);
    void backward(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target);
    void train(const Eigen::MatrixXd& input, const Eigen::MatrixXd& target, int epochs);
    Eigen::MatrixXd predict(const Eigen::MatrixXd& input);


    // private:
    std::vector<Eigen::MatrixXd> weights;
    std::vector<Eigen::MatrixXd> biases;
    std::vector<Eigen::MatrixXd> d_weights;
    std::vector<Eigen::MatrixXd> d_biases;
    activation::ActivationType activationType_;
    double learningRate;

    Eigen::MatrixXd activate(const Eigen::MatrixXd& X);
    Eigen::MatrixXd activateDeriv(const Eigen::MatrixXd& X);   
    double computeLoss(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);
    Eigen::MatrixXd computeLossDeriv(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);
    void updateWeights();
        
    std::vector<Eigen::MatrixXd> layerInputs;
    std::vector<Eigen::MatrixXd> layerOutputs;

};

} // namespace neural_network

#endif // OPTIMIZATION_NN_HPP