#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include <Eigen/Dense>
#include <vector>

class NeuralNetwork
{
    public:
        NeuralNetwork(std::vector<unsigned int> neuron_layer_num, double learning_rate);

        void forward_prop(Eigen::RowVectorXd& input);
        void backward_prop(Eigen::RowVectorXd& output);
        void eval_err(Eigen::RowVectorXd& output);
        void update_weights();
        void print_weights() const;
        void train(std::vector<Eigen::RowVectorXd*> input_data, std::vector<Eigen::RowVectorXd*> output_data);
        int predict(Eigen::RowVectorXd& input);

        std::vector<unsigned int> neuron_layer_num;
        std::vector<Eigen::RowVectorXd*> layers;
        std::vector<Eigen::RowVectorXd*> deltas;
        std::vector<Eigen::MatrixXd*> weights;
        double learning_rate;
};

#endif