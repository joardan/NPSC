#ifndef ACTIVATION_HPP
#define ACTIVATION_HPP

#include <Eigen/Dense>

class ActivationFunction
{
    public:
        virtual ~ActivationFunction() = default;
        // Pure virtual functions for the activation function and its derivative
        virtual void activate(Eigen::RowVectorXd& input) const = 0;
        virtual Eigen::RowVectorXd derivative(Eigen::RowVectorXd& input) const = 0;
};

// Derived class for ReLU activation function
class ReLU : public ActivationFunction
{
    public:
        void activate(Eigen::RowVectorXd& input) const override;
        Eigen::RowVectorXd derivative(Eigen::RowVectorXd& input) const override;
};

// Derived class for Sigmoid activation function
class Sigmoid : public ActivationFunction
{
    public:
        void activate(Eigen::RowVectorXd& input) const override;
        Eigen::RowVectorXd derivative(Eigen::RowVectorXd& input) const override;
};

#endif