#include <iostream>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Dense>


Eigen::MatrixXf relu(const Eigen::MatrixXf& input)
{
    return input.array().max(0.0);
}

Eigen::MatrixXf softmax(const Eigen::MatrixXf& input)
{
    Eigen::MatrixXf expValues = (input.array() - input.maxCoeff()).exp();
    double sum = expValues.sum();
    return expValues / sum;
}

int main()
{
    Eigen::MatrixXf input(1, 10); // Example: 1 sample, 10 classes
    // Populate input with some values

    // Compute softmax
    Eigen::MatrixXf output1 = relu(input);
    Eigen::MatrixXf output2 = softmax(input);
    std::cout << "relu: " << output1 << "\nsigmoid: " << output2;
}