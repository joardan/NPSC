#ifndef CNN_FUNCTIONS_HPP
#define CNN_FUNCTIONS_HPP

#include <Eigen/Dense>
#include <vector>

class ConvolutionalLayer;
class MaxPoolingLayer;
class FullyConnectedLayer;

double computeAccuracy(const std::vector<std::vector<Eigen::MatrixXd*>>& test_data, 
                       const std::vector<std::vector<Eigen::MatrixXd*>>& test_labels, 
                       ConvolutionalLayer& conv_layer, 
                       MaxPoolingLayer& pool_layer, 
                       FullyConnectedLayer& fc_layer);

void save_model(const ConvolutionalLayer& conv_layer, const FullyConnectedLayer& fc_layer, const std::string& file_path);

void train(std::vector<std::vector<Eigen::MatrixXd*>>& training_data, std::vector<std::vector<Eigen::MatrixXd*>>& target, 
           ConvolutionalLayer& conv_layer, MaxPoolingLayer& pool_layer, FullyConnectedLayer& fc_layer, 
           double learning_rate, int epochs, 
           const std::vector<std::vector<Eigen::MatrixXd*>>& test_data, 
           const std::vector<std::vector<Eigen::MatrixXd*>>& test_labels);

void load_model(ConvolutionalLayer& conv_layer, FullyConnectedLayer& fc_layer, const std::string& file_path);

void get_confusion_matrix(const std::vector<int>& predictions, 
                            std::vector<std::vector<Eigen::MatrixXd*>> test_labels, 
                            int num_classes);

std::vector<int> collect_predictions(const std::vector<std::vector<Eigen::MatrixXd*>>& test_data, 
                                    ConvolutionalLayer& conv_layer, 
                                    MaxPoolingLayer& pool_layer, 
                                    FullyConnectedLayer& fc_layer);

int predict(ConvolutionalLayer& conv_layer, MaxPoolingLayer& pool_layer, FullyConnectedLayer& fc_layer,
            std::vector<std::vector<Eigen::MatrixXd*>>& image_matrices);

void displayMenu();

#endif