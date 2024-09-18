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

// MAIN FILE FOR TRAINING AND SAVING MODEL
/*
int main()
{
    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/train-images.idx3-ubyte", 60000, 784);

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
*/

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


int main(int argc, char* argv[])
{
    // Initialize layers
    ConvolutionalLayer conv_layer(28, 28, 1, 5, 42);
    MaxPoolingLayer pool_layer(24, 24, 42, 2);
    FullyConnectedLayer fc_layer(pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height, 10);
    std::vector<std::vector<Eigen::MatrixXd*>> image_matrices;

    load_model(conv_layer, fc_layer, "./trained_model_data3");
    bool running = true;
    bool preprocessed = false;

    if (argc != 2)
    {
        printf("usage: Image_processing.out <Image_Path>\n");
        return -1;
    }
    image_processor processor(argv[1]);

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

    while (running) {
        displayMenu();
        int choice;
        std::cin >> choice;

        switch (choice) {
            case 1:
            {
                if (preprocessed == false)
                {
                    processor.process_demo();
                    image_matrices = processor.convert_to_eigen();
                    preprocessed = true;
                }
                break;
            }
            case 2:
            {
                if (preprocessed == false)
                {
                    processor.process_demo_lite();
                    image_matrices = processor.convert_to_eigen();
                    preprocessed = true;
                }
                break;
            }
            case 3:
            {
                if (preprocessed == false)
                {
                    processor.process();
                    image_matrices = processor.convert_to_eigen();
                    preprocessed = true;
                }
                break;
            }
            case 4:
            {
                if(image_matrices.empty())
                {
                    std::cout << "No image found" << std::endl;
                    break;
                }
                else if(preprocessed == false)
                {
                    std::cout << "Process the image first" << std::endl;
                }
                else
                {
                    predict(conv_layer, pool_layer, fc_layer, image_matrices);
                    break;
                }
            }
            case 5:
                running = false;
                std::cout << "Exitted\n";
                break;
            default:
                std::cout << "Invalid choice. Please try again\n";
        }
    }

    return 0;
}

// trained_model_data is for 3x3 filter 32 filter_num
// trained_model_data2-5 is for 5x5 filter with 42 filter_num