#include <vector>
#include "neural_network.hpp"


NeuralNetwork::NeuralNetwork(double learning_rate): learning_rate(learning_rate), onehot(10) {}

NeuralNetwork::~NeuralNetwork()
{
    for (Layer* layer : layers)
    {
        delete layer;
    }
}

void NeuralNetwork::add_Layer(unsigned int input_size, unsigned int output_size, const std::string& activation, const std::string& initialiser)
{
    layers.push_back(new Layer(input_size, output_size, activation, initialiser));
}

void NeuralNetwork::forward_prop(const Eigen::RowVectorXd& input)
{
    // Propagate forward through all layers
    for (size_t i = 0; i < layers.size(); ++i)
    {
        layers[i]->forward((i == 0) ? input : layers[i-1]->output);
        if(i != layers.size() - 1)
            layers[i]->activation->activate(layers[i]->output);
    }
    // Apply softmax to the output of the last layer
    layers.back()->output = softmax(layers.back()->output);
}

void NeuralNetwork::backward_prop(const Eigen::RowVectorXd& target)
{
    // Calculate Loss
    Eigen::RowVectorXd target_encoded = onehot.encode(static_cast<int>(target(0)));
    layers.back()->deltas = (target_encoded - layers.back()->output);

    // Calculate gradients
    for (int i = layers.size() - 2; i >= 0; --i)
    {
        Eigen::RowVectorXd layer_deltas = layers[i + 1]->deltas * layers[i + 1]->weights.transpose();
        layer_deltas.array() *= layers[i]->activation->derivative(layers[i]->output).array();
        layers[i]->deltas = layer_deltas;
    }

    // Update weights
    for (int i = 0; i < layers.size(); ++i)
    {
        layers[i]->update_weights(learning_rate);
    }
}

void NeuralNetwork::train(const std::vector<Eigen::RowVectorXd*>& input_data, const std::vector<Eigen::RowVectorXd*>& output_data)
{
    for (size_t i = 0; i < input_data.size(); ++i)
    {
        forward_prop(*input_data[i]);
        backward_prop(*output_data[i]);
    }
}

int NeuralNetwork::predict(const Eigen::RowVectorXd& input)
{
    forward_prop(input);
    return max_arg(layers.back()->output);
}

Eigen::RowVectorXd NeuralNetwork::softmax(const Eigen::RowVectorXd& x)
{
    Eigen::RowVectorXd shifted_x = x.array() - x.maxCoeff();
    Eigen::RowVectorXd exp_values = shifted_x.array().exp();
    return exp_values / exp_values.sum();
}

int NeuralNetwork::max_arg(const Eigen::RowVectorXd& x)
{
    Eigen::Index arg;
    x.maxCoeff(&arg);
    return arg;
}