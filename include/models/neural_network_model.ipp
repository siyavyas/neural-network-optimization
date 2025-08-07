#include <models/neural_network_model.hpp>

namespace nn_model {
template<typename Features, typename Target>
NeuralNetworkModel<Features, Target>::NeuralNetworkModel() {
    std::cout << "Creating IrisDataManager..." << std::endl;
    dataManager = std::make_unique<IrisDataManager>();
    std::cout << "Creating MLModel..." << std::endl;
    model = std::make_unique<MLModel<Features, Target>>(std::move(dataManager));
}

template<typename Features, typename Target>
void NeuralNetworkModel<Features, Target>::LoadDataStream(std::istream& stream) {
    std::cout << "Loading dataManager data..." << std::endl;
    dataManager->loadDataStream(stream);
}

template<typename Features, typename Target>
void NeuralNetworkModel<Features, Target>::SplitData(double trainRatio) {
    dataManager->splitData(trainRatio);
}

template<typename Features, typename Target>
void NeuralNetworkModel<Features, Target>::Train(int epochs) {
    model->train(epochs);
}

template<typename Features, typename Target>
std::string NeuralNetworkModel<Features, Target>::Predict(const Features& input) {
    return model->predict(input);
}

} // namespace optimization