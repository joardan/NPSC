#include "activation.hpp"

// ReLU Activation Functions
void ReLU::activate(Eigen::RowVectorXd& input) const
{
    input = input.array().max(0.0);
}

Eigen::RowVectorXd ReLU::derivative(Eigen::RowVectorXd& input) const
{
    return (input.array() > 0.0).cast<double>();
}


// Sigmoid Activation Functions
void Sigmoid::activate(Eigen::RowVectorXd& input) const
{
    input = 1.0 / (1.0 + (-input.array()).exp());
}

Eigen::RowVectorXd Sigmoid::derivative(Eigen::RowVectorXd& input) const
{
    Eigen::RowVectorXd sigmoid_x = 1.0 / (1.0 + (-input.array()).exp());
    return sigmoid_x.array() * (1.0 - sigmoid_x.array());
}
