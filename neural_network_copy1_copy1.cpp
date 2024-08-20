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


class Layer
{
    public:
        virtual void forward(const Eigen::RowVectorXd&);
}

class DenseLayer: public Layer
{
public:
    unsigned int input_size;
    unsigned int output_size;

    Eigen::MatrixXd weights;
    Eigen::RowVectorXd bias;
    Eigen::RowVectorXd input;
    Eigen::RowVectorXd output;
    Eigen::RowVectorXd deltas;

    Layer(unsigned int input_size, unsigned int output_size)
        : input_size(input_size), output_size(output_size)
    {
        weights = Eigen::MatrixXd::Zero(input_size, output_size);
        bias = Eigen::RowVectorXd::Ones(output_size);
        deltas = Eigen::RowVectorXd::Zero(output_size);
        initializeHe(weights);
    }

    void forward(const Eigen::RowVectorXd& input)
    {
        this->input = input;
        //std::cout << "weights row and col " << weights.rows() << " rows, " << weights.cols() << " cols.\n";
        //std::cout << "input row and col " << input.rows() << " rows, " << input.cols() << " cols.\n";
        output = input * weights;
        output += bias;
    }

    void updateWeights(double learning_rate)
    {
        Eigen::MatrixXd delta_weight = learning_rate * input.transpose() * deltas;
        weights += delta_weight;
        bias += learning_rate * deltas;
    }

private:
    void initializeHe(Eigen::MatrixXd& weights)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dis(0, sqrt(2.0 / weights.rows()));
        for (int i = 0; i < weights.rows(); ++i)
        {
            for (int j = 0; j < weights.cols(); ++j)
            {
                weights(i, j) = dis(gen);
            }
        }
    }

    void initializeXavier(Eigen::MatrixXd& weights)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(-sqrt(6.0 / (weights.rows() + weights.cols())), sqrt(6.0 / (weights.rows() + weights.cols())));
        for (int i = 0; i < weights.rows(); ++i)
        {
            for (int j = 0; j < weights.cols(); ++j)
            {
                weights(i, j) = dis(gen);
            }
        }
    }
};

class NeuralNetwork
{
public:
    NeuralNetwork(double learning_rate): learning_rate(learning_rate) {}

    ~NeuralNetwork()
    {
        for (Layer* layer : layers)
        {
            delete layer;
        }
    }

    void addLayer(unsigned int input_size, unsigned int output_size)
    {
        layers.push_back(new Layer(input_size, output_size));
    }

    void forward_prop(const Eigen::RowVectorXd& input)
    {
        layers.front()->forward(input);
        //std::cout << "first_forward_ok\n";
        for (size_t i = 1; i < layers.size(); ++i)
        {
            layers[i]->forward(layers[i - 1]->output);
            if(i != layers.size() - 1)
            {
                activationFunction(layers[i]->output);
            }
        }
        //std::cout << "forward_no_act_f_ok\n";
        // Apply softmax to the output of the last layer
        layers.back()->output = softmax(layers.back()->output);
    }

    void backward_prop(const Eigen::RowVectorXd& target)
    {
        OneHot onehot(10);
        Eigen::RowVectorXd target_encoded = onehot.encode(static_cast<int>(target(0)));
        // Calculate the output layer deltas
        //std::cout << "target row and col " << target_encoded.rows() << " rows, " << target_encoded.cols() << " cols.\n";
        //std::cout << "layer_output row and col " << layers.back()->getOutput().rows() << " rows, " << layers.back()->getOutput().cols() << " cols.\n";
        layers.back()->deltas = target_encoded - layers.back()->output;
        //std::cout << "set_first_delta_ok\n";
        // Backpropagate through the layers
        for (size_t i = layers.size() - 2; i > 0; --i)
        {
            Eigen::RowVectorXd previous_deltas = layers[i + 1]->deltas;
            Eigen::RowVectorXd layer_deltas = previous_deltas * layers[i + 1]->weights.transpose();
            layer_deltas.array() *= activationFunctionDerivative(layers[i]->output).array();
            layers[i]->deltas = layer_deltas;
        }
        //std::cout << "backprop_ok\n";
        // Update weights
        for (int i = 1; i <= layers.size() - 1; ++i)
        {
            layers[i]->updateWeights(learning_rate);
        }
    }

    void train(const std::vector<Eigen::RowVectorXd*>& input_data, const std::vector<Eigen::RowVectorXd*>& output_data)
    {
        for (size_t i = 0; i < input_data.size(); ++i)
        {
            forward_prop(*input_data[i]);
            //std::cout << "forward_ok\n";
            backward_prop(*output_data[i]);
        }
    }

    int predict(const Eigen::RowVectorXd& input)
    {
        forward_prop(input);
        return max_arg(layers.back()->output);
    }

private:
    double learning_rate;
    std::vector<Layer*> layers;

    void activationFunction(Eigen::RowVectorXd& input)
    {
        input = input.array().max(0.0f);
    }

    Eigen::RowVectorXd activationFunctionDerivative(const Eigen::RowVectorXd& x)
    {
        return (x.array() > 0.0f).cast<double>();
    }

    Eigen::RowVectorXd softmax(const Eigen::RowVectorXd& x)
    {
        Eigen::RowVectorXd shifted_x = x.array() - x.maxCoeff();
        Eigen::RowVectorXd exp_values = shifted_x.array().exp();
        return exp_values / exp_values.sum();
    }

    int max_arg(const Eigen::RowVectorXd& x)
    {
        Eigen::Index arg;
        x.maxCoeff(&arg);
        return arg;
    }
};

/*
void activationFunction(Eigen::RowVectorXd& input)
{
    input = 1.0 / (1.0 + (-input.array()).exp());
}

Eigen::RowVectorXd activationFunctionDerivative(const Eigen::RowVectorXd& x)
{
    Eigen::RowVectorXd sigmoid_x = 1.0 / (1.0 + (-x.array()).exp());
    return sigmoid_x.array() * (1.0 - sigmoid_x.array());
}
*/

int max_arg(const Eigen::RowVectorXd& x)
    {
        Eigen::Index arg;
        x.maxCoeff(&arg);
        return arg;
    }

double calculateAccuracy(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, NeuralNetwork& model)
{
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        int predicted_label = model.predict(*inputs[i]);
        int true_label = (*targets[i])(0);
        if (predicted_label == true_label)
        {
            correct++;
        }
    }
    std::cout << "Correct: " << correct << ", out of " << inputs.size() << std::endl;
    return static_cast<double>(correct) / inputs.size();
}

double kFoldCrossValidation(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, int k) {
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    double total_accuracy = 0.0f;
    size_t fold_size = inputs.size() / k;
    for (int i = 0; i < k; ++i) {
        NeuralNetwork model(0.0242);
        // Add layers
        model.addLayer(784, 32);
        model.addLayer(32, 16);
        model.addLayer(16, 10); // Output layer, no bias
        //std::cout << "addLayer_ok\n";

        std::vector<Eigen::RowVectorXd*> train_inputs, train_targets, test_inputs, test_targets;
        for (int j = 0; j < inputs.size(); ++j)
        {
            if (j >= i * fold_size && j < (i + 1) * fold_size)
            {
                test_inputs.push_back(inputs[indices[j]]);
                test_targets.push_back(targets[indices[j]]);
            }
            else
            {
                train_inputs.push_back(inputs[indices[j]]);
                train_targets.push_back(targets[indices[j]]);
            }
        }
        
        //std::cout << "data_preprocessing_ok\n";
        model.train(train_inputs, train_targets);
        //model.print_weights();
        double accuracy = calculateAccuracy(test_inputs, test_targets, model);
        total_accuracy += accuracy;
    }

    return total_accuracy / k;
}

int main()
{
    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/train-images.idx3-ubyte", 60000, 784);

    // Convert MNIST images to Eigen vectors
    std::vector<Eigen::RowVectorXd*> mnist_train_vectors = mnistImageToEigenVector(mnist_image_train, 60000, 784);
    std::vector<Eigen::RowVectorXd*> mnist_train_label_vectors = mnistLabelToEigenVector(mnist_label_train, 60000);

    // Train the model
    int k = 3; // Change this value as needed
    double average_accuracy = kFoldCrossValidation(mnist_train_vectors, mnist_train_label_vectors, k);
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