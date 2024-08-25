#include <vector>
#include <iostream>
#include <random>
#include <thread>
#include <Eigen/Dense>
#include "mnist2.hpp"
#include "encoder.hpp"
#include "initialiser.hpp"
#include "activation.hpp"
#include "layer.hpp"
#include "neural_network.hpp"
#include "tester.hpp"


int main()
{
    // Set threads, it's still slow
    Eigen::setNbThreads(std::thread::hardware_concurrency());
    
    unsigned char *mnist_label_test = read_mnist_label("../dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("../dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("../dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("../dataset/train-images.idx3-ubyte", 60000, 784);

    // Convert MNIST images to Eigen vectors
    std::vector<Eigen::RowVectorXd*> mnist_train_vectors = mnistImageToEigenVector(mnist_image_train, 60000, 784);
    std::vector<Eigen::RowVectorXd*> mnist_train_label_vectors = mnistLabelToEigenVector(mnist_label_train, 60000);

    // Train the model
    int k = 3; // Number of folds
    int epochs = 10; // Maximum number of epochs
    int early_stopping_patience = 2; // Early stopping patience
    NeuralNetworkTester tester;
    double average_accuracy = tester.kFoldCrossValidation(mnist_train_vectors, mnist_train_label_vectors, k, epochs, early_stopping_patience);
    std::cout << "Average Accuracy: " << average_accuracy << std::endl;
    
    // Cleanup
    delete[] mnist_label_test;
    delete[] mnist_label_train;
    delete[] mnist_image_test;
    delete[] mnist_image_train;

    // Cleanup the vectors
    for (auto& vec : mnist_train_vectors) {
        delete vec;
    }

    return 0;
}