#ifndef OPTIMIZATION_KFOLD_HPP
#define OPTIMIZATION_KFOLD_HPP

#include <Eigen/Dense>


namespace cv {

    class kFold {
    public:
        static std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> split(const Eigen::MatrixXd& data, 
        const Eigen::MatrixXd& labels, int k);
    };
    
} //namespace cv

#endif // OPTIMIZATION_KFOLD_HPP
