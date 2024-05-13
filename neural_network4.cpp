#include "neural_network2.hpp"
#include "mnist2.hpp"
#include "encoder.hpp"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>


NeuralNetwork::NeuralNetwork(std::vector<unsigned int> neuron_layer_num, float learning_rate)
{
    this->neuron_layer_num = neuron_layer_num;
    this->learning_rate = learning_rate;
    for(unsigned int i = 0; i < neuron_layer_num.size(); i++)
    {
        // Add new layer and assign it with the number of nodes specified in the argument
        if(i == neuron_layer_num.size() - 1)
        {
            layers.push_back(new Eigen::RowVectorXf(neuron_layer_num[i]));
        }
        else
        {
            layers.push_back(new Eigen::RowVectorXf(neuron_layer_num[i] + 1));
        }

        unactive_layers.push_back(new Eigen::RowVectorXf(layers.size()));
        deltas.push_back(new Eigen::RowVectorXf());

        // Set bias node value to 1
        if(i != neuron_layer_num.size()-1)
        {
            layers.back()->coeffRef(neuron_layer_num[i]) = 0.001;
            unactive_layers.back()->coeffRef(neuron_layer_num[i]) = 0.001;
        }

        // Set weight per layer and initialise with random values [-1, 1]
        if(i > 0)
        {
            if(i != neuron_layer_num.size()-1)
            {
                weights.push_back(new Eigen::MatrixXf(neuron_layer_num[i-1] + 1, neuron_layer_num[i] + 1));
                weights.back()->setRandom();
            }
            else
            {
                weights.push_back(new Eigen::MatrixXf(neuron_layer_num[i-1] + 1, neuron_layer_num[i]));
                weights.back()->setRandom();
            }
        }
    }
}

void activationFunction(Eigen::RowVectorXf& input)
{
	input.array().max(0.0f);
}

float activationFunctionDerivative(float x)
{
	return x > 0.0f ? 1.0f : 0.0f;
}

Eigen::RowVectorXf activationFunctionDerivative2(const Eigen::RowVectorXf& x)
{
    return (x.array() > 0.0f).cast<float>();
}

Eigen::RowVectorXf softmax(const Eigen::RowVectorXf& x)
{
    Eigen::RowVectorXf exp_values = x.array().exp();
    return exp_values / exp_values.sum();
}

int max_arg(const Eigen::RowVectorXf& x)
{
	Eigen::Index arg;
	x.maxCoeff(&arg);
    return arg;
}

void NeuralNetwork::forward_prop(Eigen::RowVectorXf& input)
{
    layers.front()->block(0, 0, 1, layers.front()->size()-1) = input;
    for(unsigned int i = 1; i < neuron_layer_num.size(); i++)
    {
        (*layers[i]) = (*layers[i-1]) * (*weights[i-1]);
        activationFunction(*layers[i]);
    }
    (*layers.back()) = softmax(*layers.back());
}

void NeuralNetwork::eval_err(Eigen::RowVectorXf& output)
{
	(*deltas.back()) = output - (*layers.back());
	for (unsigned int i = neuron_layer_num.size() - 2; i > 0; i--)
    {
		(*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
	}
}

void NeuralNetwork::update_weights()
{
	for (unsigned int i = 0; i < neuron_layer_num.size() - 1; i++)
	{
		if (i != neuron_layer_num.size() - 2)
		{
			for (unsigned int c = 0; c < weights[i]->cols() - 1; c++)
			{
				for (unsigned int r = 0; r < weights[i]->rows(); r++)
				{
					weights[i]->coeffRef(r, c) += learning_rate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(unactive_layers[i + 1]->coeffRef(c)) * layers[i]->coeffRef(r);
				}
			}
		}
		else
		{
			for (unsigned int c = 0; c < weights[i]->cols(); c++)
			{
				for (unsigned int r = 0; r < weights[i]->rows(); r++)
				{
					weights[i]->coeffRef(r, c) += learning_rate * deltas[i + 1]->coeffRef(c) * activationFunctionDerivative(unactive_layers[i + 1]->coeffRef(c)) * layers[i]->coeffRef(r);
				}
			}
		}
	}
}

void NeuralNetwork::backward_prop(Eigen::RowVectorXf& output)
{
    // Change this later
    OneHot onehot(10);
    Eigen::RowVectorXf output_encoded = onehot.encode(static_cast<int>(output(0)));
    eval_err(output_encoded);
	update_weights();
}

void NeuralNetwork::train(std::vector<Eigen::RowVectorXf*> input_data, std::vector<Eigen::RowVectorXf*> output_data)
{
	for (unsigned int i = 0; i < input_data.size(); i++)
	{
		forward_prop(*input_data[i]);
		std::cout << "Expected output is : " << *output_data[i] << std::endl;
        std::cout << "Output produced is : " << max_arg(*layers.back()) << std::endl;
		std::cout << "Output probability is : " << *layers.back() << std::endl;
		backward_prop(*output_data[i]);
	}
}

float calculateAccuracy(const std::vector<Eigen::RowVectorXf*>& inputs, const std::vector<Eigen::RowVectorXf*>& targets, NeuralNetwork& model)
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
    return static_cast<float>(correct) / inputs.size();
}

// Define a function to perform k-fold cross-validation
float kFoldCrossValidation(const std::vector<Eigen::RowVectorXf*>& inputs, const std::vector<Eigen::RowVectorXf*>& targets, NeuralNetwork& model, int k) {
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    float total_accuracy = 0.0f;
    size_t fold_size = inputs.size() / k;
    for (int i = 0; i < k; ++i) {
        std::vector<Eigen::RowVectorXf*> train_inputs, train_targets, test_inputs, test_targets;
        for (size_t j = 0; j < inputs.size(); ++j) {
            if (j >= i * fold_size && j < (i + 1) * fold_size) {
                test_inputs.push_back(inputs[indices[j]]);
                test_targets.push_back(targets[indices[j]]);
            } else {
                train_inputs.push_back(inputs[indices[j]]);
                train_targets.push_back(targets[indices[j]]);
            }
        }

        model.train(train_inputs, train_targets);
        float accuracy = calculateAccuracy(test_inputs, test_targets, model);
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
    std::vector<Eigen::RowVectorXf*> mnist_train_vectors = mnistImageToEigenVector(mnist_image_train, 60000, 784);
    std::vector<Eigen::RowVectorXf*> mnist_train_label_vectors = mnistLabelToEigenVector(mnist_label_train, 60000);

    NeuralNetwork n({784, 300, 100, 10}, 0.001);
    int k = 3; // Change this value as needed
    float average_accuracy = kFoldCrossValidation(mnist_train_vectors, mnist_train_label_vectors, n, k);
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