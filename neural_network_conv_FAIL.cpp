#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <Eigen/Dense>
#include "mnist2.hpp"
#include "encoder.hpp"

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
                       unsigned int filter_size, unsigned int filter_num, unsigned int stride = 1)
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

    ~ConvolutionalLayer() 
    {
        for (auto& mat : input)
            delete mat;
        for (auto& mat : output)
            delete mat;
        for (auto& mat : input_deltas)
            delete mat;
    }


    void forward(const std::vector<Eigen::MatrixXd*>& input)
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

    void backward(const std::vector<Eigen::MatrixXd*>& d_out, double learning_rate)
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


    Eigen::MatrixXd padMatrix(const Eigen::MatrixXd& input, int padRows, int padCols)
    {
        int paddedRows = input.rows() + 2 * padRows;
        int paddedCols = input.cols() + 2 * padCols;

        Eigen::MatrixXd paddedInput = Eigen::MatrixXd::Zero(paddedRows, paddedCols);
        paddedInput.block(padRows, padCols, input.rows(), input.cols()) = input;

        return paddedInput;
    }

    Eigen::MatrixXd correlate2d(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel) {
        int kernelRows = kernel.rows();
        int kernelCols = kernel.cols();
        int outputRows = input.rows() - kernelRows + 1;
        int outputCols = input.cols() - kernelCols + 1;

        Eigen::MatrixXd output(outputRows, outputCols);

        for (int i = 0; i < outputRows; ++i) {
            for (int j = 0; j < outputCols; ++j) {
                output(i, j) = (input.block(i, j, kernelRows, kernelCols).cwiseProduct(kernel)).sum();
            }
        }

        return output;
    }

    Eigen::MatrixXd correlate2d_full(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel)
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

    Eigen::MatrixXd convolve2d_full(const Eigen::MatrixXd& input, const Eigen::MatrixXd& kernel)
    {
        // Flip the kernel horizontally and vertically
        Eigen::MatrixXd flippedKernel = kernel.reverse();

        return correlate2d_full(input, flippedKernel);
    }


private:
    void applyReLU(std::vector<Eigen::MatrixXd*>& matrices)
    {
        for (auto& matrix : matrices)
        {
            *matrix = matrix->array().max(0.0);
        }
    }

    void glorot_uniform(Eigen::MatrixXd& weights)
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
};

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
                        unsigned int pool_size, unsigned int stride = 2)
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
                input[d] = new Eigen::MatrixXd(input_height, input_width);
            }
        }

        void forward(const std::vector<Eigen::MatrixXd*>& input)
        {
            this->input = input;
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

        void backward(const std::vector<Eigen::MatrixXd*>& d_out, std::vector<Eigen::MatrixXd*>& d_input)
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
};

class FullyConnectedLayer
{
public:
    unsigned int input_size;
    unsigned int output_size;

    Eigen::MatrixXd weights;
    Eigen::RowVectorXd bias;
    Eigen::MatrixXd* input;
    Eigen::MatrixXd* output;

    FullyConnectedLayer(unsigned int input_size, unsigned int output_size)
        : input_size(input_size), output_size(output_size)
    {
        weights = Eigen::MatrixXd::Zero(input_size, output_size);
        init_he(weights);

        bias = Eigen::RowVectorXd::Ones(output_size);
        output = new Eigen::MatrixXd(1, output_size);
    }

    ~FullyConnectedLayer()
    {
        delete output;
    }

    void forward(Eigen::MatrixXd* input)
    {
        this->input = input;
        *output = (*input) * weights;
        *output += bias;
        applyReLU(output);
        *output = softmax(*output);
    }

    void backward(const Eigen::MatrixXd& d_out, Eigen::MatrixXd& d_input, double learning_rate)
    {
        // Calculate gradients
        //std::cout << "d_out: " << d_out.rows() << " rows, " << d_out.cols() << " cols" << "\n";
        //std::cout << "weights: " << weights.rows() << " rows, " << weights.cols() << " cols" << "\n";
        //std::cout << "*output: " << output->rows() << " rows, " << output->cols() << " cols" << "\n";
        //std::cout << "*input: " << input->rows() << " rows, " << input->cols() << " cols" << "\n";

        // Calculate deltas (derivative of the loss w.r.t. output)
        Eigen::MatrixXd layer_deltas = d_out;
        layer_deltas.array() *= derivative(*output).array(); // Element-wise multiplication with ReLU derivative

        // Backpropagate the error to the input of this layer
        d_input = layer_deltas * weights.transpose(); // This computes the gradient to pass to the previous layer

        // Update weights and bias using gradient descent
        weights -= learning_rate * input->transpose() * layer_deltas; // Gradient descent step for weights
        bias -= learning_rate * layer_deltas.colwise().sum(); // Gradient descent step for bias


    }

private:
    Eigen::RowVectorXd softmax(const Eigen::RowVectorXd& x)
    {
        // Shift the input vector by its maximum value for numerical stability
        Eigen::RowVectorXd shifted_x = x.array() - x.maxCoeff();
        
        // Apply the exponential function element-wise
        Eigen::RowVectorXd exp_values = shifted_x.array().exp();
        
        // Compute the sum of all the exponentials
        double sum_exp_values = exp_values.sum();
        
        // Divide each exponential value by the sum to get the probabilities
        return exp_values / sum_exp_values;
    }


    void applyReLU(Eigen::MatrixXd* matrix)
    {
        *matrix = matrix->array().max(0.0);
    }

    Eigen::MatrixXd derivative(Eigen::MatrixXd& input) const
    {
        return (input.array() > 0.0).cast<double>();
    }

    void init_he(Eigen::MatrixXd& weights)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0, sqrt(2.0 / weights.rows()));
        for (int i = 0; i < weights.rows(); ++i)
        {
            for (int j = 0; j < weights.cols(); ++j)
            {
                weights(i, j) = dis(gen);
            }
        }
    }
};


double computeAccuracy(const std::vector<std::vector<Eigen::MatrixXd*>>& test_data, 
                       const std::vector<std::vector<Eigen::MatrixXd*>>& test_labels, 
                       ConvolutionalLayer& conv_layer, 
                       MaxPoolingLayer& pool_layer, 
                       FullyConnectedLayer& fc_layer) {
    int correct_predictions = 0;

    for (size_t i = 0; i < test_data.size(); ++i) {
        // Forward pass through the convolutional layer
        conv_layer.forward(test_data[i]);
        
        // Forward pass through the max-pooling layer
        pool_layer.forward(conv_layer.output);

        // Flatten the output from the pooling layer
        Eigen::MatrixXd flattened = Eigen::MatrixXd::Zero(1, pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height);
        for (unsigned int d = 0; d < pool_layer.output_depth; ++d) {
            Eigen::Map<Eigen::RowVectorXd> flattened_slice(pool_layer.output[d]->data(), pool_layer.output[d]->size());
            flattened.block(0, d * pool_layer.output[d]->size(), 1, pool_layer.output[d]->size()) = flattened_slice;
        }

        // Forward pass through the fully connected layer
        fc_layer.forward(&flattened);

        // Prediction (getting the class with the highest probability)
        Eigen::MatrixBase<Eigen::MatrixXd>::Index maxIndex;
        fc_layer.output->row(0).maxCoeff(&maxIndex);
        int predicted_class = static_cast<int>(maxIndex);

        // Actual label
        int actual_class = static_cast<int>((*test_labels[i][0])(0, 0));

        if (predicted_class == actual_class) {
            ++correct_predictions;
        }
    }

    return static_cast<double>(correct_predictions) / test_data.size();
}


void train(std::vector<std::vector<Eigen::MatrixXd*>>& training_data, std::vector<std::vector<Eigen::MatrixXd*>>& target, 
           ConvolutionalLayer& conv_layer, MaxPoolingLayer& pool_layer, FullyConnectedLayer& fc_layer, 
           double learning_rate, int epochs, 
           const std::vector<std::vector<Eigen::MatrixXd*>>& test_data, 
           const std::vector<std::vector<Eigen::MatrixXd*>>& test_labels)
{
    OneHot onehot(10);
    
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        int correct_predictions = 0;

        for (size_t i = 0; i < training_data.size(); ++i)
        {
            // Forward pass through the convolutional layer
            conv_layer.forward(training_data[i]);
            
            // Forward pass through the max-pooling layer
            pool_layer.forward(conv_layer.output);
            //std::cout << "pool and conv forward ok\n";
            // Flatten the output from the pooling layer
            Eigen::MatrixXd flattened = Eigen::MatrixXd::Zero(1, pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height);
            for (unsigned int d = 0; d < pool_layer.output_depth; ++d)
            {
                Eigen::Map<Eigen::RowVectorXd> flattened_slice(pool_layer.output[d]->data(), pool_layer.output[d]->size());
                flattened.block(0, d * pool_layer.output[d]->size(), 1, pool_layer.output[d]->size()) = flattened_slice;
            }
            //std::cout << "flatten ok\n";
            // Forward pass through the fully connected layer
            fc_layer.forward(&flattened);
            //std::cout << "fc_layer forward ok\n";
            
            // Prediction (getting the class with the highest probability)
            Eigen::MatrixBase<Eigen::MatrixXd>::Index maxIndex;
            fc_layer.output->row(0).maxCoeff(&maxIndex);
            int predicted_class = static_cast<int>(maxIndex);

            // Actual label
            int actual_class = static_cast<int>((*target[i][0])(0, 0));

            // Display prediction and actual label
            //std::cout << "Probability: " << fc_layer.output->row(0) << std::endl
            //          << "Prediction: " << predicted_class << " (" 
            //          << fc_layer.output->row(0)(predicted_class) * 100.0 << "%), "
            //          << "Actual Label: " << actual_class << std::endl;


            Eigen::RowVectorXd target_encoded = onehot.encode(static_cast<int>((*target[i][0])(0, 0)));
            Eigen::MatrixXd loss_grad = *fc_layer.output - target_encoded;
            //std::cout << "first delta ok\n";
            // Backward pass through the fully connected layer
            Eigen::MatrixXd d_fc_input;
            fc_layer.backward(loss_grad, d_fc_input, learning_rate);
            //std::cout << "fc backwards ok\n";
            // Reshape d_fc_input to match the expected input for the pooling layer
            std::vector<Eigen::MatrixXd*> d_pool_output(pool_layer.output_depth);
            for (unsigned int d = 0; d < pool_layer.output_depth; ++d)
            {
                d_pool_output[d] = new Eigen::MatrixXd(pool_layer.output_height, pool_layer.output_width);
                *d_pool_output[d] = Eigen::Map<Eigen::MatrixXd>(
                    d_fc_input.data() + d * pool_layer.output_height * pool_layer.output_width, 
                    pool_layer.output_height, 
                    pool_layer.output_width
                );
            }
            //std::cout << "recreate from maxpool ok\n";
            std::vector<Eigen::MatrixXd*> d_pool_input(pool_layer.input_depth);
            for (unsigned int d = 0; d < pool_layer.input_depth; ++d)
            {
                d_pool_input[d] = new Eigen::MatrixXd(pool_layer.input_height, pool_layer.input_width);
            }
            // Backward pass through the max-pooling layer
            pool_layer.backward(d_pool_output, d_pool_input); // Passing the same d_pool_input to get d_input
            //std::cout << "pool_layer backward ok\n";
            // Backward pass through the convolutional layer
            conv_layer.backward(d_pool_input, learning_rate);
            //std::cout << "conv_layer backward ok\n";
            // Clean up dynamically allocated memory
            for (auto& matrix : d_pool_input)
            {
                delete matrix;
            }
        }
        // Compute and display the test accuracy
        double test_accuracy = computeAccuracy(test_data, test_labels, conv_layer, pool_layer, fc_layer);
        std::cout << "Epoch " << epoch + 1 << " - Test Accuracy: " << test_accuracy << std::endl;
    }
}

int main()
{
    //std::cout << "start ok\n";
    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/train-images.idx3-ubyte", 60000, 784);

    // Convert MNIST images to Eigen matrices
    std::vector<Eigen::MatrixXd*> mnist_train_matrices = mnistImageToEigenMatrix(mnist_image_train, 60000, 28, 28);
    std::vector<Eigen::RowVectorXd*> mnist_train_label_vectors = mnistLabelToEigenVector(mnist_label_train, 60000);
    std::vector<Eigen::MatrixXd*> mnist_test_matrices = mnistImageToEigenMatrix(mnist_image_test, 10000, 28, 28);
    std::vector<Eigen::RowVectorXd*> mnist_test_label_vectors = mnistLabelToEigenVector(mnist_label_test, 10000);
    //std::cout << "convert data ok\n";

    // Initialize layers
    ConvolutionalLayer conv_layer(28, 28, 1, 3, 32);
    //std::cout << "conv create ok\n";
    MaxPoolingLayer pool_layer(26, 26, 32, 2);
    //std::cout << "pool create ok\n";
    FullyConnectedLayer fc_layer(pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height, 10);
    //std::cout << "create fc_layers ok\n";

    // Prepare training data and labels
    std::vector<std::vector<Eigen::MatrixXd*>> training_data;
    std::vector<std::vector<Eigen::MatrixXd*>> labels;
    //std::cout << "data grab ok\n";
    for (size_t i = 0; i < mnist_train_matrices.size(); ++i)
    {
        std::vector<Eigen::MatrixXd*> data_point = {mnist_train_matrices[i]};
        
        // Convert label to a matrix format compatible with network output
        Eigen::MatrixXd* label_matrix = new Eigen::MatrixXd(1, 1);
        (*label_matrix)(0, 0) = (*mnist_train_label_vectors[i])(0);  // Assuming labels are single values
        
        training_data.push_back(data_point);
        labels.push_back({label_matrix});
    }
    
    std::vector<std::vector<Eigen::MatrixXd*>> test_data;
    std::vector<std::vector<Eigen::MatrixXd*>> test_labels;
    for (size_t i = 0; i < mnist_test_matrices.size(); ++i)
    {
        std::vector<Eigen::MatrixXd*> data_point = {mnist_test_matrices[i]};
        
        // Convert label to a matrix format compatible with network output
        Eigen::MatrixXd* label_matrix = new Eigen::MatrixXd(1, 1);
        (*label_matrix)(0, 0) = (*mnist_test_label_vectors[i])(0);  // Assuming labels are single values
        
        test_data.push_back(data_point);
        test_labels.push_back({label_matrix});
    }
    
    double learning_rate = 0.005;
    int epochs = 10;
    //std::cout << "ready train ok\n";
    train(training_data, labels, conv_layer, pool_layer, fc_layer, learning_rate, epochs, test_data, test_labels);
    double accuracy = computeAccuracy(test_data, test_labels, conv_layer, pool_layer, fc_layer);
    std::cout << accuracy << std::endl;
    // Clean up dynamically allocated memory
    for (auto& label : labels)
    {
        delete label[0];
    }

    return 0;
}