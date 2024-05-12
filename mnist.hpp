#ifndef MNIST_H
#define MNIST_H

void displayMNISTImage(unsigned char* image, int rows, int cols);
int reverseInt(int i);
unsigned char* read_mnist_label(const std::string& file_path, int num_items);
unsigned char** read_mnist_image(const std::string& file_path, int num_items, int image_size);
Eigen::MatrixXf mnistImageToEigenMatrix(unsigned char** mnist_image, int num_items, int image_size);

#endif