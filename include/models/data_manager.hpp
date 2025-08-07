#ifndef OPTIMIZATION_DATA_MANAGER_HPP
#define OPTIMIZATION_DATA_MANAGER_HPP

#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <random>
#include <algorithm>

namespace optimization {

template<typename Features, typename Target>
class DataManager {
public:
    DataManager() = default;
    DataManager(const std::string& filePath, bool hasTarget);
    //virtual ~DataManager() = default;

    virtual void loadDataStream(std::istream& stream) = 0;
    virtual void splitData(double trainRatio) = 0;
    virtual const Eigen::MatrixXd& getTrainFeatures() const = 0;
    virtual const Eigen::MatrixXd& getTrainTargets() const = 0;
    virtual const Eigen::MatrixXd& getTestFeatures() const = 0;
    virtual const Eigen::MatrixXd& getTestTargets() const = 0;

protected:
    std::vector<std::pair<Features, Target>> parsedData;
    Eigen::MatrixXd trainFeatures;
    Eigen::MatrixXd trainTargets;
    Eigen::MatrixXd testFeatures;
    Eigen::MatrixXd testTargets;
};

} // namespace optimization

#include <models/data_manager.ipp>

#endif // OPTIMIZATION_DATA_MANAGER_HPP