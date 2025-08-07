#ifndef OPTIMIZATION_DATA_MANAGER_IPP
#define OPTIMIZATION_DATA_MANAGER_IPP

#include <models/data_manager.hpp>

namespace optimization {

template<typename Features, typename Target>
DataManager<Features, Target>::DataManager(const std::string& filePath, bool hasTarget) 
    : _hasTarget(hasTarget) {}

template<typename Features, typename Target>
void DataManager<Features, Target>::splitData(double trainRatio) {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(parsedData.begin(), parsedData.end(), g);

    size_t trainSize = static_cast<size_t>(parsedData.size() * trainRatio);
    
    trainFeatures.resize(trainSize, Features::RowsAtCompileTime);
    trainTargets.resize(trainSize, Target::RowsAtCompileTime);
    testFeatures.resize(parsedData.size() - trainSize, Features::RowsAtCompileTime);
    testTargets.resize(parsedData.size() - trainSize, Target::RowsAtCompileTime);

    for (size_t i = 0; i < parsedData.size(); ++i) {
        if (i < trainSize) {
            trainFeatures.row(i) = parsedData[i].first;
            trainTargets.row(i) = parsedData[i].second;
        } else {
            testFeatures.row(i - trainSize) = parsedData[i].first;
            testTargets.row(i - trainSize) = parsedData[i].second;
        }
    }
}

template<typename Features, typename Target>
const Eigen::MatrixXd& DataManager<Features, Target>::getTrainFeatures() const {
    return trainFeatures;
}

template<typename Features, typename Target>
const Eigen::MatrixXd& DataManager<Features, Target>::getTrainTargets() const {
    return trainTargets;
}

template<typename Features, typename Target>
const Eigen::MatrixXd& DataManager<Features, Target>::getTestFeatures() const {
    return testFeatures;
}

template<typename Features, typename Target>
const Eigen::MatrixXd& DataManager<Features, Target>::getTestTargets() const {
    return testTargets;
}

} // namespace optimization

#endif // OPTIMIZATION_DATA_MANAGER_IPP