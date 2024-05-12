#include "neural_network2.hpp"
#include "encoder.hpp"
#include "mnist.hpp"
#include <iostream>
#include <fstream>

int main()
{
    NeuralNetwork n({784, 784, 784, 10}, 0.005);

    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/train-images.idx3-ubyte", 60000, 784);
    Eigen::MatrixXf mnist_train_matrix = mnistImageToEigenMatrix(mnist_image_train, 60000, 784);
    Eigen::MatrixXf mnist_test_matrix = mnistImageToEigenMatrix(mnist_image_train, 10000, 784);

    std::vector<Eigen::RowVectorXf*> in_dat, out_dat;
    genData("test");
    ReadCSV("test-in", in_dat);
    ReadCSV("test-out", out_dat);
    n.train(in_dat, out_dat);
    return 0;
}