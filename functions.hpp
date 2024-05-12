#ifndef FUNCTIONS_HPP
#define FUNCTIONS_HPP

#include <Eigen/Dense>

Eigen::MatrixXf relu(const Eigen::MatrixXf& input);
Eigen::MatrixXf softmax(const Eigen::MatrixXf& input);

#endif