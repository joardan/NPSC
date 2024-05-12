#ifndef ENCODER_HPP
#define ENCODER_HPP

#include <Eigen/Dense>

class OneHot
{
private:
    unsigned char num_classes;

public:
    // Constructor
    OneHot(unsigned char classes);

    // One-hot encoder function
    Eigen::Matrix<unsigned char, 1, Eigen::Dynamic> encode(int label) const;

    // One-hot decoder function
    unsigned char decode(const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>& encoded) const;
};

#endif