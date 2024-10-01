#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <Eigen/Dense>
#include "conv_max_pool.hpp"

MaxPoolingLayer::MaxPoolingLayer(unsigned int input_width, unsigned int input_height, unsigned int input_depth,
                                unsigned int pool_size, unsigned int stride)
                                : input_width(input_width), input_height(input_height), input_depth(input_depth),
                                pool_size(pool_size), stride(stride)
{
    output_width = (input_width - pool_size) / stride + 1;
    output_height = (input_height - pool_size) / stride + 1;
    output_depth = input_depth;

    output.resize(input_depth);
    input.resize(input_depth);
    max_indices.resize(input_depth);
    for (unsigned int d = 0; d < input_depth; ++d)
    {
        output[d] = new Eigen::MatrixXd(output_height, output_width);
    }
}
MaxPoolingLayer::~MaxPoolingLayer()
{
    for(auto& out : output)
    {
        delete out;
    }
}
void MaxPoolingLayer::forward(const std::vector<Eigen::MatrixXd*>& input)
{
    for (unsigned int d = 0; d < input_depth; ++d)
    {
        output[d]->setZero();
        max_indices[d].resize(output_height * output_width);

        for (unsigned int i = 0; i < output_height; ++i)
        {
            for (unsigned int j = 0; j < output_width; ++j)
            {
                Eigen::MatrixXd patch = input[d]->block(i * stride, j * stride, pool_size, pool_size);
                Eigen::Index maxRow, maxCol;
                double maxVal = patch.maxCoeff(&maxRow, &maxCol);
                (*output[d])(i, j) = maxVal;
                max_indices[d][i * output_width + j] = {i * stride + maxRow, j * stride + maxCol};
            }
        }
    }
}

void MaxPoolingLayer::backward(const std::vector<Eigen::MatrixXd*>& d_out, std::vector<Eigen::MatrixXd*>& d_input)
{
    d_input.resize(input_depth);
    for (unsigned int d = 0; d < input_depth; ++d)
    {
        d_input[d]->setZero();

        for (unsigned int i = 0; i < output_height; ++i)
        {
            for (unsigned int j = 0; j < output_width; ++j)
            {
                auto max_idx = max_indices[d][i * output_width + j];
                (*d_input[d])(max_idx.first, max_idx.second) += (*d_out[d])(i, j);
            }
        }
    }
}
