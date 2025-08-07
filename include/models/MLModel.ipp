#include <models/MLModel.hpp>

namespace ml_model {

template<typename Features, typename Target>
MLModel<Features, Target>::MLModel(DataManager<Features, Target>& dataManager)
    : dataManager(dataManager) {
        std::cout << "Creating Neural Network..." << std::endl;
        nn = std::make_unique<neural_network::NeuralNetwork>(
            std::vector<int>{4, 10, 3},
            activation::ActivationType::Sigmoid,
            0.01
        );
        std::cout << "Neural Network created." << std::endl;
}

template<typename Features, typename Target>
void MLModel<Features, Target>::train(int epochs, bool verbose) {
    if (verbose) {
        std::cout << "Training Neural Network..." << std::endl;
    }
    nn->train(dataManager->getTrainFeatures(), dataManager->getTrainTargets(), epochs);
}

template<typename Features, typename Target>
std::string MLModel<Features, Target>::predict(const Features& input) {
    Eigen::MatrixXd inputMatrix = input.transpose();
    Eigen::MatrixXd output = nn->predict(inputMatrix);

    int predictedClass = 0;
    output.row(0).maxCoeff(&predictedClass);

    return decodeLabel(predictedClass);
}

template<typename Features, typename Target>
std::string MLModel<Features, Target>::decodeLabel(int encodedLabel) {
    switch(encodedLabel) {
        case 0: return "Iris-setosa";
        case 1: return "Iris-versicolor";
        case 2: return "Iris-virginica";
        default: return "Unknown";
    }
}

template<typename Features, typename Target>
std::vector<std::string> MLModel<Features, Target>::decodeLabels(const std::vector<double>& encodedLabels) {
    std::vector<std::string> decodedLabels;
    for (const double& label : encodedLabels) {
        decodedLabels.push_back(decodeLabel(static_cast<int>(label)));
    }
    return decodedLabels;
}

} // namespace ml_model