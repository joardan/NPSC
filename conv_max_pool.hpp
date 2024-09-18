#ifndef MAX_POOL_HPP
#define MAX_POOL_HPP

#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <Eigen/Dense>

class MaxPoolingLayer
{
    public:
        unsigned int input_width;
        unsigned int input_height;
        unsigned int input_depth;
        unsigned int pool_size;
        unsigned int stride;
        unsigned int output_width;
        unsigned int output_height;
        unsigned int output_depth;

        std::vector<Eigen::MatrixXd*> input;
        std::vector<Eigen::MatrixXd*> output;
        std::vector<std::vector<std::pair<unsigned int, unsigned int>>> max_indices;

        MaxPoolingLayer(unsigned int input_width, unsigned int input_height, unsigned int input_depth,
                        unsigned int pool_size, unsigned int stride = 2);
        void forward(const std::vector<Eigen::MatrixXd*>& input);
        void backward(const std::vector<Eigen::MatrixXd*>& d_out, std::vector<Eigen::MatrixXd*>& d_input);
};

#endif