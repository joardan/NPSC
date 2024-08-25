#include "layer.hpp"


// Initialise layers with weights, bias, etc.
Layer::Layer(unsigned int input_size, unsigned int output_size, const std::string& activation, const std::string& initialiser)
    : input_size(input_size), output_size(output_size)
{
    weights = Eigen::MatrixXd::Zero(input_size, output_size);
    bias = Eigen::RowVectorXd::Ones(output_size);
    deltas = Eigen::RowVectorXd::Zero(output_size);
    this->activation = assign_activation_function(activation);
    init.init(weights, initialiser);
}

Layer::~Layer()
{
    delete activation;
}

// Calculate input forward with weights and biases
void Layer::forward(const Eigen::RowVectorXd& input)
{
    this->input = input;
    output = input * weights;
    output += bias;
}

// Update bias and weights according to deltas and learning rate
void Layer::update_weights(double learning_rate)
{
    Eigen::MatrixXd delta_weight = learning_rate * input.transpose() * deltas;
    weights += delta_weight;
    bias += learning_rate * deltas;
}

// Assign activation function for the layer
ActivationFunction* Layer::assign_activation_function(const std::string& activation)
{
    if (activation == "relu")
    {
        return new ReLU();
    }
    else if (activation == "sigmoid")
    {
        return new Sigmoid();
    }
    throw std::invalid_argument("Unknown activation function: " + activation);
}
