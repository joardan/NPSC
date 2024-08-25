#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <thread>
#include <Eigen/Dense>
#include "mnist2.hpp"
#include "encoder.hpp"


class Initialiser
{
    public:
        void init(Eigen::MatrixXd& weights, const std::string& initialiser)
        {
            if(initialiser == "he")
            {
                initialiseHe(weights);
            }
            else if(initialiser == "xavier")
            {
                initialiseXavier(weights);
            }
        }

    private:
        void initialiseHe(Eigen::MatrixXd& weights)
        {
            // Create random generator for a distribution for He init
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

        void initialiseXavier(Eigen::MatrixXd& weights)
        {
            // Create random generator for a distribution for Xavier init
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

// Base class for Activation Functions
class ActivationFunction
{
    public:
        virtual ~ActivationFunction() = default;

        // Pure virtual functions for the activation function and its derivative
        virtual void activate(Eigen::RowVectorXd& input) const = 0;
        virtual Eigen::RowVectorXd derivative(Eigen::RowVectorXd& input) const = 0;
};

// Derived class for ReLU activation function
class ReLU : public ActivationFunction
{
    public:
        void activate(Eigen::RowVectorXd& input) const override
        {
            input = input.array().max(0.0);
        }

        Eigen::RowVectorXd derivative(Eigen::RowVectorXd& input) const override
        {
            return (input.array() > 0.0).cast<double>();
        }
};

// Derived class for Sigmoid activation function
class Sigmoid : public ActivationFunction
{
    public:
        void activate(Eigen::RowVectorXd& input) const override
        {
            input = 1.0 / (1.0 + (-input.array()).exp());
        }

        Eigen::RowVectorXd derivative(Eigen::RowVectorXd& input) const override
        {
            Eigen::RowVectorXd sigmoid_x = 1.0 / (1.0 + (-input.array()).exp());
            return sigmoid_x.array() * (1.0 - sigmoid_x.array());
        }
};


class Layer
{
public:
    unsigned int input_size;
    unsigned int output_size;

    Eigen::MatrixXd weights;
    Eigen::RowVectorXd bias;
    Eigen::RowVectorXd input;
    Eigen::RowVectorXd output;
    Eigen::RowVectorXd deltas;
    Initialiser init;
    ActivationFunction* activation;

    // Initialise layers with weights, bias, etc.
    Layer(unsigned int input_size, unsigned int output_size, const std::string& activation = "relu", const std::string& initialiser = "he")
        : input_size(input_size), output_size(output_size)
    {
        weights = Eigen::MatrixXd::Zero(input_size, output_size);
        bias = Eigen::RowVectorXd::Ones(output_size);
        deltas = Eigen::RowVectorXd::Zero(output_size);
        this->activation = assign_activation_function(activation);
        init.init(weights, initialiser);
    }

    ~Layer()
    {
        delete activation;
    }

    void forward(const Eigen::RowVectorXd& input)
    {
        this->input = input;
        //std::cout << "weights row and col " << weights.rows() << " rows, " << weights.cols() << " cols.\n";
        //std::cout << "input row and col " << input.rows() << " rows, " << input.cols() << " cols.\n";
        output = input * weights;
        output += bias;
    }

    void update_weights(double learning_rate)
    {
        Eigen::MatrixXd delta_weight = learning_rate * input.transpose() * deltas;
        weights += delta_weight;
        bias += learning_rate * deltas;
    }

    private:

        ActivationFunction* assign_activation_function(const std::string& activation)
        {
            if (activation == "relu")
            {
                return new ReLU();
            }
            else if (activation == "sigmoid")
            {
                return new Sigmoid();
            }
            throw std::invalid_argument("Unknown activation function: " + activation);
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

    void add_Layer(unsigned int input_size, unsigned int output_size)
    {
        layers.push_back(new Layer(input_size, output_size));
    }

    void forward_prop(const Eigen::RowVectorXd& input)
    {
        // Propagate forward through all layers
        for (size_t i = 0; i < layers.size(); ++i)
        {
            layers[i]->forward((i == 0) ? input : layers[i-1]->output);
            if(i != layers.size() - 1)
                layers[i]->activation->activate(layers[i]->output);
        }
        // Apply softmax to the output of the last layer
        layers.back()->output = softmax(layers.back()->output);
    }

    void backward_prop(const Eigen::RowVectorXd& target)
    {
        OneHot onehot(10);
        // Calculate Loss
        Eigen::RowVectorXd target_encoded = onehot.encode(static_cast<int>(target(0)));
        layers.back()->deltas = (target_encoded - layers.back()->output);

        // Calculate gradients
        for (int i = layers.size() - 2; i >= 0; --i)
        {
            Eigen::RowVectorXd layer_deltas = layers[i + 1]->deltas * layers[i + 1]->weights.transpose();
            layer_deltas.array() *= layers[i]->activation->derivative(layers[i]->output).array();
            layers[i]->deltas = layer_deltas;
        }

        // Update weights
        for (int i = 0; i < layers.size(); ++i)
        {
            layers[i]->update_weights(learning_rate);
        }
    }

    void train(const std::vector<Eigen::RowVectorXd*>& input_data, const std::vector<Eigen::RowVectorXd*>& output_data)
    {
        for (size_t i = 0; i < input_data.size(); ++i)
        {
            forward_prop(*input_data[i]);
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

int max_arg(const Eigen::RowVectorXd& x)
    {
        Eigen::Index arg;
        x.maxCoeff(&arg);
        return arg;
    }

double calculate_accuracy(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, NeuralNetwork& model)
{
    // Calculate accuracy by going through test set
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

double kFoldCrossValidation(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, int k, int epochs, int early_stopping_patience) {
    // Create Indices for Kfold
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    double total_accuracy = 0.0;
    size_t fold_size = inputs.size() / k;

    for (int i = 0; i < k; ++i) {
        NeuralNetwork model(0.0242);
        // Add layers
        model.add_Layer(784, 64);
        model.add_Layer(64, 32);
        model.add_Layer(32, 16);
        model.add_Layer(16, 10);

        // Split the data into training and testing using generated index
        std::vector<Eigen::RowVectorXd*> train_inputs, train_targets, test_inputs, test_targets;
        for (int j = 0; j < inputs.size(); ++j) {
            if (j >= i * fold_size && j < (i + 1) * fold_size) {
                test_inputs.push_back(inputs[indices[j]]);
                test_targets.push_back(targets[indices[j]]);
            } else {
                train_inputs.push_back(inputs[indices[j]]);
                train_targets.push_back(targets[indices[j]]);
            }
        }

        // Early stopping variables
        int best_epoch = 0;
        double best_accuracy = 0.0;
        int patience_counter = 0;

        for (int epoch = 1; epoch <= epochs; ++epoch) {
            // Train on whole dataset each epoch
            model.train(train_inputs, train_targets);

            // Evaluate on the testing data
            double test_accuracy = calculate_accuracy(test_inputs, test_targets, model);
            std::cout << "Fold " << i + 1 << " - Epoch " << epoch << " - Test Accuracy: " << test_accuracy << std::endl;

            // Early stopping check
            if (test_accuracy > best_accuracy) {
                best_accuracy = test_accuracy;
                best_epoch = epoch;
                patience_counter = 0;
            } else {
                patience_counter++;
            }

            if (patience_counter >= early_stopping_patience) {
                std::cout << "Early stopping at epoch " << epoch << " with best test accuracy of " << best_accuracy << " at epoch " << best_epoch << "." << std::endl;
                break;
            }
        }

        total_accuracy += best_accuracy;
    }

    return total_accuracy / k;
}

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

    double average_accuracy = kFoldCrossValidation(mnist_train_vectors, mnist_train_label_vectors, k, epochs, early_stopping_patience);
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