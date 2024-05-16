#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <Eigen/Dense>

class OneHot
{
private:
    int num_classes;

public:
    // Constructor
    OneHot(int classes);

    // One-hot encoder function
    Eigen::RowVectorXd encode(int label) const;

    // One-hot decoder function
    int decode(const Eigen::RowVectorXd& encoded) const;
};

#endif