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
    Eigen::RowVectorXf encode(int label) const;

    // One-hot decoder function
    int decode(const Eigen::RowVectorXf& encoded) const;
};

#endif