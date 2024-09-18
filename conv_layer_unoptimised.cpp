#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include "conv_layer_unoptimised.hpp"

ConvolutionalLayer::ConvolutionalLayer(unsigned int input_width, unsigned int input_height, unsigned int input_depth,
                    unsigned int filter_size, unsigned int filter_num, unsigned int stride)
    : input_width(input_width), input_height(input_height), input_depth(input_depth),
        filter_size(filter_size), filter_num(filter_num), stride(stride)
{
    output_width = (input_width - filter_size) / stride + 1;
    output_height = (input_height - filter_size) / stride + 1;

    filters.resize(filter_num);
    for (auto& filter : filters)
    {
        filter.resize(input_depth, Eigen::MatrixXd(filter_size, filter_size));
        for (auto& depth_slice : filter)
        {
            glorot_uniform(depth_slice);
        }
    }

    bias = Eigen::RowVectorXd::Zero(filter_num);

    output.resize(filter_num);
    for (unsigned int f = 0; f < filter_num; ++f)
    {
        output[f] = new Eigen::MatrixXd(output_height, output_width);
    }

    // Preallocate the input_deltas matrices
    input_deltas.resize(input_depth);
    for (unsigned int d = 0; d < input_depth; ++d)
    {
        input_deltas[d] = new Eigen::MatrixXd(input_height, input_width);
    }
}

ConvolutionalLayer::~ConvolutionalLayer() 
{
    for (auto& mat : input)
        delete mat;
    for (auto& mat : output)
        delete mat;
    for (auto& mat : input_deltas)
        delete mat;
}


void ConvolutionalLayer::forward(const std::vector<Eigen::MatrixXd*>& input)
{
    this->input = input;
    for (unsigned int f = 0; f < filter_num; ++f)
    {
        output[f]->setZero();
        for (unsigned int i = 0; i < output_height; ++i)
        {
            for (unsigned int j = 0; j < output_width; ++j)
            {
                for (unsigned int d = 0; d < input_depth; ++d)
                {
                    (*output[f])(i, j) +=
                        (input[d]->block(i * stride, j * stride, filter_size, filter_size).array() *
                            filters[f][d].array())
                            .sum();
                }
                (*output[f])(i, j) += bias(f);
            }
        }
    }
    applyReLU(output);
}

void ConvolutionalLayer::backward(const std::vector<Eigen::MatrixXd*>& d_out, double learning_rate)
{
    // Initialize gradient matrices
    std::vector<std::vector<Eigen::MatrixXd>> filter_gradients(filter_num, std::vector<Eigen::MatrixXd>(input_depth));

    for (unsigned int f = 0; f < filter_num; ++f)
    {
        for (unsigned int d = 0; d < input_depth; ++d)
        {
            filter_gradients[f][d] = Eigen::MatrixXd::Zero(filter_size, filter_size);
        }
    }
    Eigen::RowVectorXd bias_grad = Eigen::RowVectorXd::Zero(filter_num);

    // Compute gradients for filters and bias
    for (unsigned int f = 0; f < filter_num; ++f)
    {
        for (unsigned int i = 0; i < output_height; ++i)
        {
            for (unsigned int j = 0; j < output_width; ++j)
            {
                for (unsigned int d = 0; d < input_depth; ++d)
                {
                    // Filter gradient
                    filter_gradients[f][d] += input[d]->block(i * stride, j * stride, filter_size, filter_size) * (*d_out[f])(i, j);
                }
                // Bias gradient
                bias_grad(f) += (*d_out[f])(i, j);
            }
        }
    }

    // Update filters and bias
    for (unsigned int f = 0; f < filter_num; ++f)
    {
        for (unsigned int d = 0; d < input_depth; ++d)
        {
            filters[f][d] -= learning_rate * filter_gradients[f][d];
        }
    }
    bias -= learning_rate * bias_grad;

    // Propagate the gradient to the previous layer (input deltas)
    for (unsigned int d = 0; d < input_depth; ++d)
    {
        input_deltas[d]->setZero();

        for (unsigned int f = 0; f < filter_num; ++f)
        {
            *input_deltas[d] += convolve2d_full(*d_out[f], filters[f][d]);
        }
    }
}

Eigen::MatrixXd ConvolutionalLayer::padMatrix(const Eigen::MatrixXd& input, int padRows, int padCols)
{
    int paddedRows = input.rows() + 2 * padRows;
    int paddedCols = input.cols() + 2 * padCols;

    Eigen::MatrixXd paddedInput = Eigen::MatrixXd::Zero(paddedRows, paddedCols);
    paddedInput.block(padRows, padCols, input.rows(), input.cols()) = input;

    return paddedInput;
}

Eigen::MatrixXd ConvolutionalLayer::correlate2d_full(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel)
{
    int kernelRows = kernel.rows();
    int kernelCols = kernel.cols();
    
    // Padding size
    int padRows = kernelRows - 1;
    int padCols = kernelCols - 1;

    // Pad the input matrix
    Eigen::MatrixXd paddedInput = padMatrix(input, padRows, padCols);

    // Output matrix size
    int outputRows = paddedInput.rows() - kernelRows + 1;
    int outputCols = paddedInput.cols() - kernelCols + 1;

    Eigen::MatrixXd output(outputRows, outputCols);

    for (int i = 0; i < outputRows; ++i)
    {
        for (int j = 0; j < outputCols; ++j)
        {
            output(i, j) = (paddedInput.block(i, j, kernelRows, kernelCols).cwiseProduct(kernel)).sum();
        }
    }

    return output;
}

Eigen::MatrixXd ConvolutionalLayer::convolve2d_full(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel)
{
    // Flip the kernel horizontally and vertically
    Eigen::MatrixXd flippedKernel = kernel.reverse();

    return correlate2d_full(input, flippedKernel);
}



void ConvolutionalLayer::applyReLU(std::vector<Eigen::MatrixXd*>& matrices)
{
    for (auto& matrix : matrices)
    {
        *matrix = matrix->array().max(0.0);
    }
}

void ConvolutionalLayer::glorot_uniform(Eigen::MatrixXd& weights)
{
    double limit = std::sqrt(6.0 / (weights.rows() + weights.cols()));
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-limit, limit);

    for (int i = 0; i < weights.rows(); ++i)
    {
        for (int j = 0; j < weights.cols(); ++j)
        {
            weights(i, j) = dis(gen);
        }
    }
}