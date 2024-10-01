#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include "mnist2.hpp"
#include "encoder.hpp"
#include "multi_numbers_processing.hpp"
#include "conv_layer_unoptimised.hpp"
#include "conv_max_pool.hpp"
#include "conv_fc_layer.hpp"
#include "conv_function.hpp"

double computeAccuracy(const std::vector<std::vector<Eigen::MatrixXd*>>& test_data, 
                       const std::vector<std::vector<Eigen::MatrixXd*>>& test_labels, 
                       ConvolutionalLayer& conv_layer, 
                       MaxPoolingLayer& pool_layer, 
                       FullyConnectedLayer& fc_layer)
{
    int correct_predictions = 0;

    for (size_t i = 0; i < test_data.size(); ++i)
    {
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

        if (predicted_class == actual_class)
        {
            ++correct_predictions;
        }
    }

    return static_cast<double>(correct_predictions) / test_data.size();
}

void save_model(const ConvolutionalLayer& conv_layer, const FullyConnectedLayer& fc_layer, const std::string& file_path)
{
    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs.is_open())
    {
        std::cerr << "Failed to open file for saving model.\n";
        return;
    }

    // Save convolutional filters
    for (const auto& filters : conv_layer.filters)
    {
        for (const auto& filter : filters) {
            for (int i = 0; i < filter.rows(); ++i)
            {
                for (int j = 0; j < filter.cols(); ++j)
                {
                    ofs.write(reinterpret_cast<const char*>(&filter(i, j)), sizeof(double));
                }
            }
        }
    }

    // Save convolutional biases
    for (int i = 0; i < conv_layer.bias.size(); ++i)
    {
        ofs.write(reinterpret_cast<const char*>(&conv_layer.bias(i)), sizeof(double));
    }

    // Save fully connected weights
    for (int i = 0; i < fc_layer.weights.rows(); ++i)
    {
        for (int j = 0; j < fc_layer.weights.cols(); ++j)
        {
            ofs.write(reinterpret_cast<const char*>(&fc_layer.weights(i, j)), sizeof(double));
        }
    }

    // Save fully connected biases
    for (int i = 0; i < fc_layer.bias.size(); ++i)
    {
        ofs.write(reinterpret_cast<const char*>(&fc_layer.bias(i)), sizeof(double));
    }

    ofs.close();
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

            // Flatten the output from the pooling layer
            Eigen::MatrixXd flattened = Eigen::MatrixXd::Zero(1, pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height);
            for (unsigned int d = 0; d < pool_layer.output_depth; ++d)
            {
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
            int actual_class = static_cast<int>((*target[i][0])(0, 0));

            Eigen::RowVectorXd target_encoded = onehot.encode(static_cast<int>((*target[i][0])(0, 0)));
            Eigen::MatrixXd loss_grad = *fc_layer.output - target_encoded;

            // Backward pass through the fully connected layer
            Eigen::MatrixXd d_fc_input;
            fc_layer.backward(loss_grad, d_fc_input, learning_rate);

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

            std::vector<Eigen::MatrixXd*> d_pool_input(pool_layer.input_depth);
            for (unsigned int d = 0; d < pool_layer.input_depth; ++d)
            {
                d_pool_input[d] = new Eigen::MatrixXd(pool_layer.input_height, pool_layer.input_width);
            }
            // Backward pass through the max-pooling layer
            pool_layer.backward(d_pool_output, d_pool_input); // Passing the same d_pool_input to get d_input
            // Backward pass through the convolutional layer
            conv_layer.backward(d_pool_input, learning_rate);

            for(auto& matrix : d_pool_output)
            {
                delete matrix;
            }
            for(auto& matrix : d_pool_input)
            {
                delete matrix;
            }
        }
        // Compute and display the test accuracy
        double test_accuracy = computeAccuracy(test_data, test_labels, conv_layer, pool_layer, fc_layer);
        std::cout << "Epoch " << epoch + 1 << " - Test Accuracy: " << test_accuracy << std::endl;
    }
    save_model(conv_layer, fc_layer, "./trained_model_data5");
}

void load_model(ConvolutionalLayer& conv_layer, FullyConnectedLayer& fc_layer, const std::string& file_path)
{
    std::ifstream ifs(file_path, std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Failed to open file for loading model.\n";
        return;
    }

    // Load convolutional filters
    for (auto& filters : conv_layer.filters) {
        for (auto& filter : filters) {
            for (int i = 0; i < filter.rows(); ++i) {
                for (int j = 0; j < filter.cols(); ++j) {
                    ifs.read(reinterpret_cast<char*>(&filter(i, j)), sizeof(double));
                }
            }
        }
    }

    // Load convolutional biases
    for (int i = 0; i < conv_layer.bias.size(); ++i) {
        ifs.read(reinterpret_cast<char*>(&conv_layer.bias(i)), sizeof(double));
    }

    // Load fully connected weights
    for (int i = 0; i < fc_layer.weights.rows(); ++i) {
        for (int j = 0; j < fc_layer.weights.cols(); ++j) {
            ifs.read(reinterpret_cast<char*>(&fc_layer.weights(i, j)), sizeof(double));
        }
    }

    // Load fully connected biases
    for (int i = 0; i < fc_layer.bias.size(); ++i) {
        ifs.read(reinterpret_cast<char*>(&fc_layer.bias(i)), sizeof(double));
    }

    ifs.close();
}

void get_confusion_matrix(const std::vector<int>& predictions, 
                            std::vector<std::vector<Eigen::MatrixXd*>> test_labels, 
                            int num_classes)
{
    Eigen::MatrixXi confusion_matrix = Eigen::MatrixXi::Zero(num_classes, num_classes);

    // Populate the confusion matrix
    for (size_t i = 0; i < predictions.size(); ++i)
    {
        int true_label = (*test_labels[i][0])(0, 0);
        int predicted_label = predictions[i];
        confusion_matrix(true_label, predicted_label)++;
    }

    std::cout << "Confusion Matrix:\n" << confusion_matrix << std::endl;
}

std::vector<int> collect_predictions(const std::vector<std::vector<Eigen::MatrixXd*>>& test_data, 
                                    ConvolutionalLayer& conv_layer, 
                                    MaxPoolingLayer& pool_layer, 
                                    FullyConnectedLayer& fc_layer) {
    std::vector<int> predictions;

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

        // Get the predicted class (index of the maximum value in output)
        Eigen::MatrixBase<Eigen::MatrixXd>::Index maxIndex;
        fc_layer.output->row(0).maxCoeff(&maxIndex);
        int predicted_class = static_cast<int>(maxIndex);
        predictions.push_back(predicted_class);
    }

    return predictions;
}


int predict(ConvolutionalLayer& conv_layer, MaxPoolingLayer& pool_layer, FullyConnectedLayer& fc_layer,
            std::vector<std::vector<Eigen::MatrixXd*>>& image_matrices)
{
    for (size_t i = 0; i < image_matrices.size(); ++i)
    {
        // Forward pass through the convolutional layer
        conv_layer.forward(image_matrices[i]);
        
        // Forward pass through the max-pooling layer
        pool_layer.forward(conv_layer.output);

        // Flatten the output from the pooling layer
        Eigen::MatrixXd flattened = Eigen::MatrixXd::Zero(1, pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height);
        for (unsigned int d = 0; d < pool_layer.output_depth; ++d)
        {
            Eigen::Map<Eigen::RowVectorXd> flattened_slice(pool_layer.output[d]->data(), pool_layer.output[d]->size());
            flattened.block(0, d * pool_layer.output[d]->size(), 1, pool_layer.output[d]->size()) = flattened_slice;
        }

        // Forward pass through the fully connected layer
        fc_layer.forward(&flattened);

        // Prediction (getting the class with the highest probability)
        Eigen::MatrixBase<Eigen::MatrixXd>::Index maxIndex;
        fc_layer.output->row(0).maxCoeff(&maxIndex);
        int predicted_class = static_cast<int>(maxIndex);
        
        std::cout << "Prediction: " << predicted_class << std::endl;
    }

    return 0;
}

void displayMenu()
{
    std::cout << "Menu:\n";
    std::cout << "1. Process Image (This is demo version with multiple windows, just press any key to continue)\n";
    std::cout << "2. Process Image (RECOMMENDED: This is demo version with one window, just press any key to continue)\n";
    std::cout << "3. Process Image (This runs without showing any images)\n";
    std::cout << "4. Predict Image\n";
    std::cout << "5. Quit\n";
    std::cout << "Enter your choice: ";
}

// trained_model_data is for 3x3 filter 32 filter_num
// trained_model_data2-5 is for 5x5 filter with 42 filter_num