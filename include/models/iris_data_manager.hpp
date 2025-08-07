#ifndef MODELS_IRIS_DATA_MANAGER_HPP
#define MODELS_IRIS_DATA_MANAGER_HPP

#include <models/data_manager.hpp>
#include <Eigen/Dense>
#include <string>
#include <sstream>
#include <stdexcept>

namespace iris_model {

template<typename Features, typename Target>
class IrisDataManager : public DataManager<Eigen::MatrixXd, Eigen::MatrixXd> {
public:
    void loadDataStream(std::istream& stream) override;
    void splitData(double trainRatio) override;

    const Eigen::MatrixXd& getTrainFeatures() const override;
    const Eigen::MatrixXd& getTrainTargets() const override;
    const Eigen::MatrixXd& getTestFeatures() const override;
    const Eigen::MatrixXd& getTestTargets() const override;

private:
    void removeIDColumn();
    double convertIrisClassToDouble(const std::string& irisClass);
};

} // namespace iris_model

#include <models/iris_data_manager.ipp>

#endif // MODELS_IRIS_DATA_MANAGER_HPP