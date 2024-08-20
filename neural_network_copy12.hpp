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
        void eval_err(Eigen::RowVectorXd& target);
        void update_weights();
        void print_weights() const;
        void train(std::vector<Eigen::RowVectorXd*> input_data, std::vector<Eigen::RowVectorXd*> output_data);
        int predict(Eigen::RowVectorXd& input);

        std::vector<Layer*> layers;
        double learning_rate;
};


class Layer
{
    public:
        Layer(unsigned int neuron_num, unsigned int is_input);
        unsigned int neuron_num;
        unsigned int is_input;
        Eigen::MatrixXd* weights;
        Eigen::VectorXd* bias;
        Eigen::MatrixXd* deltas;
}
#endif