#ifndef OPTIMIZATION_ML_MODEL_HPP
#define OPTIMIZATION_ML_MODEL_HPP

#include <models/data_manager.hpp>
#include <models/iris_data_manager.hpp>
#include <models/neural_network_model.hpp>
#include <optimization/activation.hpp>

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <iostream>
#include <sstream>
#include <string>
#include <memory>

namespace ml_model {

template<typename Features, typename Target>
class MLModel {
public:
    MLModel() = default;
    MLModel(DataManager<Features, Target>& dataManager);
    void train(int epochs, bool verbose = false);
    std::string predict(const Features& input);

private:
    //std::unique_ptr<DataManager<Features, Target>> dataManager;
    DataManager<Features, Target>& dataManager;
    std::unique_ptr<neural_network::NeuralNetwork> nn;

    std::string decodeLabel(int encodedLabel);
    std::vector<std::string> decodeLabels(const std::vector<double>& encodedLabels);
};

} // namespace ml_model

#include <models/MLModel.ipp>

#endif // MODELS_ML_MODEL_HPP