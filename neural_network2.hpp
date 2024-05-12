#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Eigen/Dense>
#include <vector>

class NeuralNetwork
{
    public:
        NeuralNetwork(std::vector<unsigned int> neuron_layer_num, float learning_rate);

        void forward_prop(Eigen::RowVectorXf& input);
        void backward_prop(Eigen::RowVectorXf& output);
        void eval_err(Eigen::RowVectorXf& output);
        void update_weights();
        void train(std::vector<Eigen::RowVectorXf*> input_data, std::vector<Eigen::RowVectorXf*> output_data);

        std::vector<unsigned int> neuron_layer_num;
        std::vector<Eigen::RowVectorXf*> layers;
        std::vector<Eigen::RowVectorXf*> unactive_layers;
        std::vector<Eigen::RowVectorXf*> deltas;
        std::vector<Eigen::MatrixXf*> weights;
        float learning_rate;
};

#endif