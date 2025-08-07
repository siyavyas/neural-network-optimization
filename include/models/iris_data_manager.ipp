#include <models/data_manager.hpp>
#include <models/iris_data_manager.hpp>

namespace iris_model {

template<typename Features, typename Target>
void IrisDataManager<Features, Target>::loadDataStream(std::istream& stream) {
    std::string line;
    std::getline(stream, line); // skip header

    while (std::getline(stream, line)) {
        std::istringstream ss(line);
        std::string val;
        std::vector<double> row;

        while (std::getline(ss, val, ',')) {
            if (val == "Iris-setosa" || val == "Iris-versicolor" || val == "Iris-virginica") {
                row.push_back(convertIrisClassToDouble(val));
            } else {
                row.push_back(std::stod(val));
            }
        }

        Eigen::MatrixXd features;
        Eigen::MatrixXd target = Eigen::MatrixXd::Zero(1, 3);
        features << row[1], row[2], row[3], row[4];
        target(static_cast<int>(row[5])) = 1.0;

        parsedData.emplace_back(features, target);
    }

    removeIDColumn();
}

template<typename Features, typename Target>
void IrisDataManager<Features, Target>::splitData(double trainRatio) {
    DataManager<Eigen::MatrixXd, Eigen::MatrixXd>::splitData(trainRatio);
}

template<typename Features, typename Target>
void IrisDataManager<Features, Target>::removeIDColumn() {
    for (auto& dataPair : this->parsedData) {
        Eigen::MatrixXd& features = dataPair.first;
        features = features.block(0, 1, features.rows(), features.cols() - 1);
    }
}

template<typename Features, typename Target>
double IrisDataManager<Features, Target>::convertIrisClassToDouble(const std::string& irisClass) {
    if (irisClass == "Iris-setosa") return 0.0;
    if (irisClass == "Iris-versicolor") return 1.0;
    if (irisClass == "Iris-virginica") return 2.0;
    throw std::runtime_error("Unknown Iris class: " + irisClass);
}

template<typename Features, typename Target>
const Eigen::MatrixXd& IrisDataManager<Features, Target>::getTrainFeatures() const {
    return trainFeatures;
}

template<typename Features, typename Target>
const Eigen::MatrixXd& IrisDataManager<Features, Target>::getTrainTargets() const {
    return trainTargets;
}

template<typename Features, typename Target>
const Eigen::MatrixXd& IrisDataManager<Features, Target>::getTestFeatures() const {
    return testFeatures;
}

template<typename Features, typename Target>
const Eigen::MatrixXd& IrisDataManager<Features, Target>::getTestTargets() const {
    return testTargets;
}

} // namespace iris_model