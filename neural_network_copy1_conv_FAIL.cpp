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

// Convolutional Layer
class ConvLayer {
public:
    unsigned int num_filters;
    unsigned int filter_size;
    unsigned int stride;
    unsigned int input_height;
    unsigned int input_width;
    unsigned int output_height;
    unsigned int output_width;

    std::vector<Eigen::MatrixXd> filters;
    std::vector<Eigen::MatrixXd> filter_deltas;
    Eigen::MatrixXd d_input;
    std::vecotr<Eigen::MatrixXd> input;
    std::vector<Eigen::MatrixXd> output;

    ConvLayer(unsigned int num_filters, unsigned int filter_size, unsigned int stride, unsigned int input_height, unsigned int input_width)
        : num_filters(num_filters), filter_size(filter_size), stride(stride), input_height(input_height), input_width(input_width)
    {
        output_height = (input_height - filter_size) / stride + 1;
        output_width = (input_width - filter_size) / stride + 1;

        filters.resize(num_filters);
        filter_deltas.resize(num_filters);
        for (unsigned int i = 0; i < num_filters; ++i) {
            filters[i] = Eigen::MatrixXd::Zero(filter_size, filter_size);
            glorot_uniform(filters[i], filter_size);
            filter_deltas[i] = Eigen::MatrixXd::Zero(filter_size, filter_size);
        }
    }

    void forward(const Eigen::MatrixXd& input)
    {
        this->input = input;
        output.clear();

        for (unsigned int f = 0; f < num_filters; ++f) {
            Eigen::MatrixXd feature_map(output_height, output_width);

            for (unsigned int i = 0; i < output_height; ++i) {
                for (unsigned int j = 0; j < output_width; ++j) {
                    feature_map(i, j) = (input.block(i * stride, j * stride, filter_size, filter_size).array() * filters[f].array()).sum();
                    activationFunction(feature_map);
                }
            }
            output.push_back(feature_map);
        }
    }

    void backward(const std::vector<Eigen::MatrixXd>& d_out)
    {
        filter_deltas.assign(num_filters, Eigen::MatrixXd::Zero(filter_size, filter_size));
        d_input = Eigen::MatrixXd::Zero(input_height, input_width);

        for (unsigned int f = 0; f < num_filters; ++f)
        {
            for (unsigned int i = 0; i < output_height; ++i)
            {
                for (unsigned int j = 0; j < output_width; ++j)
                {
                    // Calculate the gradient for the filter
                    filter_deltas[f] += d_out[f](i, j) * input.block(i * stride, j * stride, filter_size, filter_size);

                    // Propagate the error to the input
                    d_input.block(i * stride, j * stride, filter_size, filter_size) += d_out[f](i, j) * filters[f];
                }
            }
        }
    }

    void updateWeights(double learning_rate)
    {
        for (unsigned int f = 0; f < num_filters; ++f) {
            filters[f] += learning_rate * filter_deltas[f];
        }
    }

    void glorot_uniform(Eigen::MatrixXd& weights, int filter_size)
    {
        // Compute the limit
        double limit = std::sqrt(3.0 / filter_size);
        
        // Random number generation
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-limit, limit);
        
        // Fill the weight matrix
        for (int i = 0; i < weights.rows(); ++i) {
            for (int j = 0; j < weights.cols(); ++j) {
                weights(i, j) = dis(gen);
            }
        }
    }

    void activationFunction(Eigen::MatrixXd& input)
    {
        input = input.array().max(0.0f);
    }

    Eigen::RowVectorXd activationFunctionDerivative(const Eigen::MatrixXd& x)
    {
        return (x.array() > 0.0f).cast<double>();
    }
};

// MaxPooling Layer
class MaxPoolingLayer {
public:
    unsigned int pool_size;
    unsigned int stride;
    unsigned int input_height;
    unsigned int input_width;
    unsigned int output_height;
    unsigned int output_width;

    std::vector<Eigen::MatrixXd> input;
    std::vector<Eigen::MatrixXd> output;
    std::vector<Eigen::MatrixXd> max_mask;

    MaxPoolingLayer(unsigned int pool_size, unsigned int stride, unsigned int input_height, unsigned int input_width)
        : pool_size(pool_size), stride(stride), input_height(input_height), input_width(input_width) {
        
        output_height = (input_height - pool_size) / stride + 1;
        output_width = (input_width - pool_size) / stride + 1;
    }

    void forward(const std::vector<Eigen::MatrixXd>& input) {
        this->input = input;
        output.clear();
        max_mask.clear();

        for (const auto& feature_map : input) {
            Eigen::MatrixXd pooled(output_height, output_width);
            Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(input_height, input_width);

            for (unsigned int i = 0; i < output_height; ++i) {
                for (unsigned int j = 0; j < output_width; ++j) {
                    Eigen::MatrixXd region = feature_map.block(i * stride, j * stride, pool_size, pool_size);
                    double max_val = region.maxCoeff();
                    pooled(i, j) = max_val;

                    for (unsigned int k = 0; k < pool_size; ++k) {
                        for (unsigned int l = 0; l < pool_size; ++l) {
                            if (region(k, l) == max_val) {
                                mask(i * stride + k, j * stride + l) = 1.0;
                                break;
                            }
                        }
                    }
                }
            }
            output.push_back(pooled);
            max_mask.push_back(mask);
        }
    }

    std::vector<Eigen::MatrixXd> backward(const std::vector<Eigen::MatrixXd>& d_out)
    {
        std::vector<Eigen::MatrixXd> d_input(input.size(), Eigen::MatrixXd::Zero(input_height, input_width));

        for (size_t f = 0; f < input.size(); ++f)
        {
            for (unsigned int i = 0; i < output_height; ++i)
            {
                for (unsigned int j = 0; j < output_width; ++j)
                {
                    // Propagate the gradient to the input based on the max mask
                    d_input[f].block(i * stride, j * stride, pool_size, pool_size) +=
                        max_mask[f].block(i * stride, j * stride, pool_size, pool_size) * d_out[f](i, j);
                }
            }
        }
        return d_input;
    }
};

// Fully Connected Layer
class FullyConnectedLayer {
public:
    unsigned int input_size;
    unsigned int output_size;

    Eigen::MatrixXd weights;
    Eigen::RowVectorXd bias;
    Eigen::RowVectorXd input;
    Eigen::RowVectorXd output;
    Eigen::RowVectorXd deltas;

    FullyConnectedLayer(unsigned int input_size, unsigned int output_size)
        : input_size(input_size), output_size(output_size) {
        weights = Eigen::MatrixXd::Zero(input_size, output_size);
        bias = Eigen::RowVectorXd::Ones(output_size);
        deltas = Eigen::RowVectorXd::Zero(output_size);
        initializeHe(weights);
    }

    void forward(const Eigen::RowVectorXd& input) {
        this->input = input;
        output = input * weights;
        output += bias;
    }

    void updateWeights(double learning_rate) {
        Eigen::MatrixXd delta_weight = learning_rate * input.transpose() * deltas;
        weights += delta_weight;
        bias += learning_rate * deltas;
    }

private:
    void initializeHe(Eigen::MatrixXd& weights) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0, sqrt(2.0 / weights.rows()));
        for (int i = 0; i < weights.rows(); ++i) {
            for (int j = 0; j < weights.cols(); ++j) {
                weights(i, j) = dis(gen);
            }
        }
    }
};

// Neural Network
class NeuralNetwork {
public:
    NeuralNetwork(double learning_rate) : learning_rate(learning_rate) {}

    ~NeuralNetwork() {
        for (auto& layer : conv_layers) delete layer;
        for (auto& layer : pool_layers) delete layer;
        for (auto& layer : fc_layers) delete layer;
    }

    void addConvLayer(unsigned int num_filters, unsigned int filter_size, unsigned int stride, unsigned int input_height, unsigned int input_width) {
        conv_layers.push_back(new ConvLayer(num_filters, filter_size, stride, input_height, input_width));
    }

    void addPoolingLayer(unsigned int pool_size, unsigned int stride, unsigned int input_height, unsigned int input_width) {
        pool_layers.push_back(new MaxPoolingLayer(pool_size, stride, input_height, input_width));
    }

    void addFullyConnectedLayer(unsigned int input_size, unsigned int output_size) {
        fc_layers.push_back(new FullyConnectedLayer(input_size, output_size));
    }

    void forward_prop(const Eigen::MatrixXd& input)
    {
        conv_layers[0]->forward(input);
        for (size_t i = 1; i < conv_layers.size(); ++i) {
            pool_layers[i - 1]->forward(conv_layers[i - 1]->output);
            conv_layers[i]->forward(pool_layers[i - 1]->output[0]);
        }

        // Pooling for the last conv layer
        pool_layers.back()->forward(conv_layers.back()->output);

        // Flatten all the feature maps into a single vector
        std::vector<Eigen::MatrixXd>& pooled_output = pool_layers.back()->output;
        Eigen::RowVectorXd flatten(pooled_output.size() * pooled_output[0].size());

        for (size_t i = 0; i < pooled_output.size(); ++i)
        {
            flatten.segment(i * pooled_output[0].size(), pooled_output[0].size()) = Eigen::Map<Eigen::RowVectorXd>(pooled_output[i].data(), pooled_output[i].size());
        }

        // Forward through the fully connected layer
        fc_layers.front()->forward(flatten);

        for (size_t i = 1; i < fc_layers.size(); ++i) {
            fc_layers[i]->forward(fc_layers[i - 1]->output);
            activationFunction(fc_layers[i]->output);
        }

        fc_layers.back()->output = softmax(fc_layers.back()->output);
    }

    void backward_prop(const Eigen::RowVectorXd& target)
    {
        OneHot onehot(10);
        Eigen::RowVectorXd target_encoded = onehot.encode(static_cast<int>(target(0)));

        // Compute output layer deltas
        fc_layers.back()->deltas = target_encoded - fc_layers.back()->output;

        // Backpropagate through fully connected layers
        for (int i = fc_layers.size() - 2; i >= 0; --i)
        {
            Eigen::RowVectorXd layer_deltas = fc_layers[i + 1]->deltas * fc_layers[i + 1]->weights.transpose();
            layer_deltas.array() *= activationFunctionDerivative(fc_layers[i]->output).array();
            fc_layers[i]->deltas = layer_deltas;
        }

        // Update fully connected layers' weights
        for (int i = 0; i < fc_layers.size(); ++i) {
            fc_layers[i]->updateWeights(learning_rate);
        }

        // Backpropagate through the pooling and convolutional layers
        std::vector<Eigen::MatrixXd> d_out;

        // Unflatten the deltas for the last FC layer and prepare for conv backprop
        d_out.resize(conv_layers.back()->num_filters);
        for (unsigned int f = 0; f < conv_layers.back()->num_filters; ++f)
        {
            d_out[f] = Eigen::Map<Eigen::MatrixXd>(fc_layers.front()->deltas.data() + f * pool_layers.back()->output[0].size(), pool_layers.back()->output[0].rows(), pool_layers.back()->output[0].cols());
        }

        // Backprop through pooling and conv layers
        for (int i = conv_layers.size() - 1; i >= 0; --i)
        {
            std::vector<Eigen::MatrixXd> pool_deltas = pool_layers[i]->backward(d_out);
            conv_layers[i]->backward(pool_deltas);
            d_out = pool_deltas;  // Pass the deltas to the previous layer
        }

        // Update conv layers' weights
        for (int i = 0; i < conv_layers.size(); ++i)
        {
            conv_layers[i]->updateWeights(learning_rate);
        }
    }

    void train(const std::vector<Eigen::MatrixXd*>& input_data, const std::vector<Eigen::RowVectorXd*>& output_data) {
        for (size_t i = 0; i < input_data.size(); ++i) {
            forward_prop(*input_data[i]);
            backward_prop(*output_data[i]);
        }
    }

    int predict(const Eigen::MatrixXd& input) {
        forward_prop(input);
        return max_arg(fc_layers.back()->output);
    }

private:
    double learning_rate;
    std::vector<ConvLayer*> conv_layers;
    std::vector<MaxPoolingLayer*> pool_layers;
    std::vector<FullyConnectedLayer*> fc_layers;

    void activationFunction(Eigen::RowVectorXd& input) {
        input = input.array().max(0.0f);
    }

    Eigen::RowVectorXd activationFunctionDerivative(const Eigen::RowVectorXd& x) {
        return (x.array() > 0.0f).cast<double>();
    }

    Eigen::RowVectorXd softmax(const Eigen::RowVectorXd& x) {
        Eigen::RowVectorXd shifted_x = x.array() - x.maxCoeff();
        Eigen::RowVectorXd exp_values = shifted_x.array().exp();
        return exp_values / exp_values.sum();
    }

    int max_arg(const Eigen::RowVectorXd& x) {
        Eigen::Index arg;
        x.maxCoeff(&arg);
        return arg;
    }
};

double calculateAccuracy(const std::vector<Eigen::MatrixXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, NeuralNetwork& model) {
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        int predicted_label = model.predict(*inputs[i]);
        int true_label = (*targets[i])(0);
        if (predicted_label == true_label) {
            correct++;
        }
    }
    std::cout << "Correct: " << correct << ", out of " << inputs.size() << std::endl;
    return static_cast<double>(correct) / inputs.size();
}

double kFoldCrossValidation(const std::vector<Eigen::MatrixXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, int k) {
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    double total_accuracy = 0.0f;
    size_t fold_size = inputs.size() / k;
    for (int i = 0; i < k; ++i) {
        NeuralNetwork model(0.001);
        // Add layers
        model.addConvLayer(8, 3, 1, 28, 28);
        model.addPoolingLayer(2, 2, 26, 26);
        model.addConvLayer(16, 3, 1, 13, 13);
        model.addPoolingLayer(2, 2, 11, 11);
        model.addFullyConnectedLayer(16 * 5 * 5, 128); // Flatten to 400 units
        model.addFullyConnectedLayer(128, 10);         // 10 output classes

        std::vector<Eigen::MatrixXd*> train_inputs, test_inputs;
        std::vector<Eigen::RowVectorXd*> train_targets, test_targets;

        for (int j = 0; j < inputs.size(); ++j) {
            if (j >= i * fold_size && j < (i + 1) * fold_size) {
                test_inputs.push_back(inputs[indices[j]]);
                test_targets.push_back(targets[indices[j]]);
            } else {
                train_inputs.push_back(inputs[indices[j]]);
                train_targets.push_back(targets[indices[j]]);
            }
        }

        model.train(train_inputs, train_targets);
        double accuracy = calculateAccuracy(test_inputs, test_targets, model);
        total_accuracy += accuracy;
    }

    return total_accuracy / k;
}

int main() {
    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/train-images.idx3-ubyte", 60000, 784);

    // Convert MNIST images to Eigen matrices
    std::vector<Eigen::MatrixXd*> mnist_train_matrices = mnistImageToEigenMatrix(mnist_image_train, 60000, 28, 28);
    std::vector<Eigen::RowVectorXd*> mnist_train_label_vectors = mnistLabelToEigenVector(mnist_label_train, 60000);

    // Train the model
    int k = 3; // Change this value as needed
    double average_accuracy = kFoldCrossValidation(mnist_train_matrices, mnist_train_label_vectors, k);
    std::cout << "Average Accuracy: " << average_accuracy << std::endl;

    // Cleanup
    delete[] mnist_label_test;
    delete[] mnist_label_train;
    delete[] mnist_image_test;
    delete[] mnist_image_train;

    // Cleanup the vectors
    for (auto& mat : mnist_train_matrices) delete mat;
    for (auto& vec : mnist_train_label_vectors) delete vec;

    return 0;
}
