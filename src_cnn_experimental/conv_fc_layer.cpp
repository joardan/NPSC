#include <random>
#include "conv_fc_layer.hpp"

FullyConnectedLayer::FullyConnectedLayer(unsigned int input_size, unsigned int output_size)
    : input_size(input_size), output_size(output_size)
{
    weights = Eigen::MatrixXd::Zero(input_size, output_size);
    init_he(weights);

    bias = Eigen::RowVectorXd::Ones(output_size);
    output = new Eigen::MatrixXd(1, output_size);
}

FullyConnectedLayer::~FullyConnectedLayer()
{
    delete output;
}

void FullyConnectedLayer::forward(Eigen::MatrixXd* input)
{
    this->input = input;
    *output = (*input) * weights;
    *output += bias;
    applyReLU(output);
    *output = softmax(*output);
}

void FullyConnectedLayer::backward(const Eigen::MatrixXd& d_out, Eigen::MatrixXd& d_input, double learning_rate)
{
    Eigen::MatrixXd layer_deltas = d_out;
    layer_deltas.array() *= derivative(*output).array(); // Element-wise multiplication with ReLU derivative

    // Backpropagate the error to the input of this layer
    d_input = layer_deltas * weights.transpose(); // This computes the gradient to pass to the previous layer

    // Update weights and bias using gradient descent
    weights -= learning_rate * input->transpose() * layer_deltas; // Gradient descent step for weights
    bias -= learning_rate * layer_deltas.colwise().sum(); // Gradient descent step for bias
}

Eigen::RowVectorXd FullyConnectedLayer::softmax(const Eigen::RowVectorXd& x)
{
    // Shift the input vector by its maximum value for numerical stability
    Eigen::RowVectorXd shifted_x = x.array() - x.maxCoeff();
    
    // Apply the exponential function element-wise
    Eigen::RowVectorXd exp_values = shifted_x.array().exp();
    
    // Compute the sum of all the exponentials
    double sum_exp_values = exp_values.sum();
    
    // Divide each exponential value by the sum to get the probabilities
    return exp_values / sum_exp_values;
}


void FullyConnectedLayer::applyReLU(Eigen::MatrixXd* matrix)
{
    *matrix = matrix->array().max(0.0);
}

Eigen::MatrixXd FullyConnectedLayer::derivative(Eigen::MatrixXd& input) const
{
    return (input.array() > 0.0).cast<double>();
}

void FullyConnectedLayer::init_he(Eigen::MatrixXd& weights)
{
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dis(0, sqrt(2.0 / weights.rows()));
    for (int i = 0; i < weights.rows(); ++i)
    {
        for (int j = 0; j < weights.cols(); ++j)
        {
            weights(i, j) = dis(gen);
        }
    }
}