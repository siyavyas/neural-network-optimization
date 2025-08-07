#ifndef NEURAL_NETWORK_MODEL_HPP
#define NEURAL_NETWORK_MODEL_HPP

#include <models/data_manager.hpp>
#include <models/iris_data_manager.hpp>
#include <models/MLModel.hpp>
#include <optimization/nn.hpp>
#include <memory>
#include <Eigen/Dense>

namespace nn_model {

template<typename Features, typename Target>
class NeuralNetworkModel {
public:
    NeuralNetworkModel();
    void LoadDataStream(std::istream& stream);
    void SplitData(double trainRatio);
    void Train(int epochs);
    std::string Predict(const Features& input);

private:
    std::unique_ptr<DataManager<Features, Target>> dataManager;
    std::unique_ptr<MLModel<Features, Target>> model;
};

} // namespace nn_model

#include <models/neural_network_model.ipp>

#endif // MODELS_NEURAL_NETWORK_MODEL_HPP