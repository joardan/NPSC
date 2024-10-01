#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include "mnist2.hpp"
#include "encoder.hpp"
#include "multi_numbers_processing.hpp"
#include "conv_layer_unoptimised.hpp"
#include "conv_max_pool.hpp"
#include "conv_fc_layer.hpp"
#include "conv_function.hpp"

int main(int argc, char* argv[])
{
    // Initialize layers
    ConvolutionalLayer conv_layer(28, 28, 1, 5, 42);
    MaxPoolingLayer pool_layer(24, 24, 42, 2);
    FullyConnectedLayer fc_layer(pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height, 10);
    std::vector<std::vector<Eigen::MatrixXd*>> image_matrices;

    load_model(conv_layer, fc_layer, "./trained_model_data5");
    bool running = true;
    bool preprocessed = false;
    
    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    std::vector<Eigen::MatrixXd*> mnist_test_matrices = mnistImageToEigenMatrix(mnist_image_test, 10000, 28, 28);
    std::vector<Eigen::RowVectorXd*> mnist_test_label_vectors = mnistLabelToEigenVector(mnist_label_test, 10000);
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
    std::vector<int> predictions;
    predictions = collect_predictions(test_data, conv_layer, pool_layer, fc_layer);
    get_confusion_matrix(predictions, test_labels, 10);

    return 0;
}

// trained_model_data is for 3x3 filter 32 filter_num
// trained_model_data2-5 is for 5x5 filter with 42 filter_num