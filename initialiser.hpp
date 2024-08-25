#ifndef INITIALISER_HPP
#define INITIALISER_HPP

#include <Eigen/Dense>

class Initialiser
{
    public:
        void init(Eigen::MatrixXd& weights, const std::string& initialiser);
    private:
        void initialiseHe(Eigen::MatrixXd& weights);
        void initialiseXavier(Eigen::MatrixXd& weights);
};

#endif