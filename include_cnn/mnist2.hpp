#ifndef MNIST_H
#define MNIST_H

#include <Eigen/Dense>

void displayMNISTImage(unsigned char* image, int rows, int cols);
int reverseInt(int i);
unsigned char* read_mnist_label(const std::string& file_path, int num_items);
unsigned char** read_mnist_image(const std::string& file_path, int num_items, int image_size);
std::vector<Eigen::RowVectorXd*> mnistImageToEigenVector(unsigned char** mnist_image, int num_items, int image_size);
std::vector<Eigen::RowVectorXd*> mnistLabelToEigenVector(unsigned char* mnist_label, int num_items);
std::vector<Eigen::MatrixXd*> mnistImageToEigenMatrix(unsigned char** mnist_image, int num_items, int image_rows, int image_cols);

#endif