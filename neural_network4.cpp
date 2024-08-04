#include "neural_network4.hpp"
#include "mnist2.hpp"
#include "encoder.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>


NeuralNetwork::NeuralNetwork(std::vector<unsigned int> neuron_layer_num, double learning_rate)
{
    this->neuron_layer_num = neuron_layer_num;
    this->learning_rate = learning_rate;
    for(unsigned int i = 0; i < neuron_layer_num.size(); i++)
    {
        // Add new layer and assign it with the number of nodes specified in the argument
        if(i == neuron_layer_num.size() - 1)
        {
            layers.push_back(new Eigen::RowVectorXd(neuron_layer_num[i]));
        }
        else
        {
            layers.push_back(new Eigen::RowVectorXd(neuron_layer_num[i] + 1));
        }

        deltas.push_back(new Eigen::RowVectorXd());

        // Set bias node value to 1
        if(i != neuron_layer_num.size()-1)
        {
            layers.back()->coeffRef(neuron_layer_num[i]) = 1;
        }

        // Set weight per layer and initialise with random values [-1, 1]
        if(i > 0)
        {
            if(i != neuron_layer_num.size()-1)
            {
                weights.push_back(new Eigen::MatrixXd(neuron_layer_num[i-1] + 1, neuron_layer_num[i] + 1));
                weights.back()->setRandom();
            }
            else
            {
                weights.push_back(new Eigen::MatrixXd(neuron_layer_num[i-1] + 1, neuron_layer_num[i]));
                weights.back()->setRandom();
            }
        }
    }
}

void NeuralNetwork::print_weights() const
{
    for (size_t i = 0; i < weights.size(); ++i)
    {
        std::cout << "Weights for layer " << i + 1 << ":\n" << *weights[i] << std::endl;
    }
}

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
    Eigen::RowVectorXd exp_values = x.array().exp();
    return exp_values / exp_values.sum();
}

int max_arg(const Eigen::RowVectorXd& x)
{
	Eigen::Index arg;
	x.maxCoeff(&arg);
    return arg;
}

void NeuralNetwork::forward_prop(Eigen::RowVectorXd& input)
{
    layers.front()->block(0, 0, 1, layers.front()->size()-1) = input;
    for(unsigned int i = 1; i < neuron_layer_num.size(); i++)
    {
        (*layers[i]) = (*layers[i-1]) * (*weights[i-1]);
        activationFunction(*layers[i]);
    }
    (*layers.back()) = softmax(*layers.back());
}

void NeuralNetwork::eval_err(Eigen::RowVectorXd& output)
{
	(*deltas.back()) = (output - (*layers.back()));
	for (unsigned int i = neuron_layer_num.size() - 2; i > 0; i--)
    {
		(*deltas[i]) = ((*deltas[i + 1]) * (weights[i]->transpose())).array() * activationFunctionDerivative(*layers[i]).array();
	}
}

void NeuralNetwork::update_weights()
{
    for (unsigned int i = 0; i < neuron_layer_num.size() - 1; i++)
    {
        weights[i]->noalias() += learning_rate * layers[i]->transpose() * (*deltas[i + 1]);
    }
}

void NeuralNetwork::backward_prop(Eigen::RowVectorXd& output)
{
    // Change this later
    OneHot onehot(10);
    Eigen::RowVectorXd output_encoded = onehot.encode(static_cast<int>(output(0)));
    eval_err(output_encoded);
	update_weights();
}

void NeuralNetwork::train(std::vector<Eigen::RowVectorXd*> input_data, std::vector<Eigen::RowVectorXd*> output_data)
{
	for (unsigned int i = 0; i < input_data.size(); i++)
	{
		forward_prop(*input_data[i]);

		//std::cout << "Expected output is : " << *output_data[i] << std::endl;
        //std::cout << "Output produced is : " << max_arg(*layers.back()) << std::endl;
		//std::cout << "Output probability is : " << *layers.back() << std::endl;

        backward_prop(*output_data[i]);
	}
}

int NeuralNetwork::predict(Eigen::RowVectorXd& input)
{
    forward_prop(input);
    return max_arg(*layers.back());
}

double calculateAccuracy(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, NeuralNetwork& model)
{
    int correct = 0;
    for (size_t i = 0; i < inputs.size(); ++i)
    {
        model.forward_prop(*inputs[i]);
        int predicted_label = max_arg(*model.layers.back());
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
        NeuralNetwork model({784, 32, 16, 10}, 0.0242);
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

    NeuralNetwork n({784, 32, 16, 10}, 0.0242);
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