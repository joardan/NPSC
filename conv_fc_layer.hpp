#ifndef FCC_HPP
#define FCC_HPP

#include <Eigen/Dense>

class FullyConnectedLayer
{
public:
    unsigned int input_size;
    unsigned int output_size;

    Eigen::MatrixXd weights;
    Eigen::RowVectorXd bias;
    Eigen::MatrixXd* input;
    Eigen::MatrixXd* output;

    FullyConnectedLayer(unsigned int input_size, unsigned int output_size);

    ~FullyConnectedLayer();

    void forward(Eigen::MatrixXd* input);

    void backward(const Eigen::MatrixXd& d_out, Eigen::MatrixXd& d_input, double learning_rate);

private:
    Eigen::RowVectorXd softmax(const Eigen::RowVectorXd& x);

    void applyReLU(Eigen::MatrixXd* matrix);

    Eigen::MatrixXd derivative(Eigen::MatrixXd& input) const;

    void init_he(Eigen::MatrixXd& weights);
};

#endif