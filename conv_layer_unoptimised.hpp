#ifndef CONV_LAYER_HPP
#define CONV_LAYER_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <Eigen/Dense>

class ConvolutionalLayer
{
public:
    unsigned int input_width;
    unsigned int input_height;
    unsigned int input_depth;
    unsigned int filter_size;
    unsigned int filter_num;
    unsigned int stride;
    unsigned int output_width;
    unsigned int output_height;

    std::vector<std::vector<Eigen::MatrixXd>> filters;  // [filter_num][input_depth]
    Eigen::RowVectorXd bias;
    std::vector<Eigen::MatrixXd*> input;
    std::vector<Eigen::MatrixXd*> output;
    std::vector<Eigen::MatrixXd*> input_deltas;

    ConvolutionalLayer(unsigned int input_width, unsigned int input_height, unsigned int input_depth,
                       unsigned int filter_size, unsigned int filter_num, unsigned int stride = 1);

    ~ConvolutionalLayer();

    void forward(const std::vector<Eigen::MatrixXd*>& input);

    void backward(const std::vector<Eigen::MatrixXd*>& d_out, double learning_rate);

    Eigen::MatrixXd padMatrix(const Eigen::MatrixXd& input, int padRows, int padCols);

    Eigen::MatrixXd correlate2d_full(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);

    Eigen::MatrixXd convolve2d_full(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel);

private:
    void applyReLU(std::vector<Eigen::MatrixXd*>& matrices);

    void glorot_uniform(Eigen::MatrixXd& weights);
};

#endif