#ifndef NEURALNETWORK_HPP
#define NEURALNETWORK_HPP

#include <Eigen/Dense>
#include "layer.hpp"
#include "encoder.hpp"

class NeuralNetwork
{
    public:
        NeuralNetwork(double learning_rate);
        ~NeuralNetwork();
        void add_Layer(unsigned int input_size, unsigned int output_size, const std::string& activation = "relu", const std::string& initialiser = "he");
        void forward_prop(const Eigen::RowVectorXd& input);
        void backward_prop(const Eigen::RowVectorXd& target);
        void train(const std::vector<Eigen::RowVectorXd*>& input_data, const std::vector<Eigen::RowVectorXd*>& output_data);
        int predict(const Eigen::RowVectorXd& input);

    private:
        OneHot onehot;
        double learning_rate;
        std::vector<Layer*> layers;
        Eigen::RowVectorXd softmax(const Eigen::RowVectorXd& x);
        int max_arg(const Eigen::RowVectorXd& x);
};

#endif