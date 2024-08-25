#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include "activation.hpp"
#include "initialiser.hpp"

class Layer
{
    public:
        unsigned int input_size;
        unsigned int output_size;

        Eigen::MatrixXd weights;
        Eigen::RowVectorXd bias;
        Eigen::RowVectorXd input;
        Eigen::RowVectorXd output;
        Eigen::RowVectorXd deltas;
        Initialiser init;
        ActivationFunction* activation;

        // Initialise layers with weights, bias, etc.
        Layer(unsigned int input_size, unsigned int output_size, const std::string& activation = "relu", const std::string& initialiser = "he");
        ~Layer();
        void forward(const Eigen::RowVectorXd& input);
        void update_weights(double learning_rate);

    private:
        ActivationFunction* assign_activation_function(const std::string& activation);
};

#endif