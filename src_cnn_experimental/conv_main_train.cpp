#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include "mnist2.hpp"
#include "encoder.hpp"
#include "multi_numbers_processing.hpp"
#include "conv_layer_unoptimised.hpp"
#include "conv_max_pool.hpp"
#include "conv_fc_layer.hpp"
#include "conv_function.hpp"

// MAIN FILE FOR TRAINING AND SAVING MODEL

int main()
{
    unsigned char *mnist_label_test = read_mnist_label("../dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("../dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("../dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("../dataset/train-images.idx3-ubyte", 60000, 784);

    // Convert MNIST images to Eigen matrices
    std::vector<Eigen::MatrixXd*> mnist_train_matrices = mnistImageToEigenMatrix(mnist_image_train, 60000, 28, 28);
    std::vector<Eigen::RowVectorXd*> mnist_train_label_vectors = mnistLabelToEigenVector(mnist_label_train, 60000);
    std::vector<Eigen::MatrixXd*> mnist_test_matrices = mnistImageToEigenMatrix(mnist_image_test, 10000, 28, 28);
    std::vector<Eigen::RowVectorXd*> mnist_test_label_vectors = mnistLabelToEigenVector(mnist_label_test, 10000);

    // Initialize layers
    ConvolutionalLayer conv_layer(28, 28, 1, 5, 42);
    MaxPoolingLayer pool_layer(24, 24, 42, 2);
    FullyConnectedLayer fc_layer(pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height, 10);

    // Prepare training data and labels
    std::vector<std::vector<Eigen::MatrixXd*>> training_data;
    std::vector<std::vector<Eigen::MatrixXd*>> labels;

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
    
    double learning_rate = 0.001;
    int epochs = 8;

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

// trained_model_data is for 3x3 filter 32 filter_num
// trained_model_data2-5 is for 5x5 filter with 42 filter_num