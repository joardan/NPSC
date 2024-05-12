#include "neural_network2.hpp"
#include "mnist2.hpp"
#include "encoder.hpp"
#include <iostream>
#include <fstream>
#include <cmath>


NeuralNetwork::NeuralNetwork(std::vector<unsigned int> neuron_layer_num, float learning_rate)
{
    this->neuron_layer_num = neuron_layer_num;
    this->learning_rate = learning_rate;
    for(unsigned int i = 0; i < neuron_layer_num.size(); i++)
    {
        if(i == neuron_layer_num.size() - 1)
            layers.push_back(new Eigen::RowVectorXf(neuron_layer_num[i]));
        else
            layers.push_back(new Eigen::RowVectorXf(neuron_layer_num[i] + 1));
        
        unactive_layers.push_back(new Eigen::RowVectorXf(layers.size()));
        deltas.push_back(new Eigen::RowVectorXf(layers.size()));

        if(i != neuron_layer_num.size()-1)
        {
            layers.back()->coeffRef(neuron_layer_num[i]) = 1.0;
            unactive_layers.back()->coeffRef(neuron_layer_num[i]) = 1.0;
        }

        if(i > 0)
        {
            if(i != neuron_layer_num.size()-1)
            {
                weights.push_back(new Eigen::MatrixXf(neuron_layer_num[i-1] + 1, neuron_layer_num[i] + 1));
                weights.back()->setRandom();
                weights.back()->col(neuron_layer_num[i]).setZero();
                weights.back()->coeffRef(neuron_layer_num[i-1], neuron_layer_num[i]) = 1.0;
            }
            else
            {
                weights.push_back(new Eigen::MatrixXf(neuron_layer_num[i-1] + 1, neuron_layer_num[i]));
                weights.back()->setRandom();
            }
        }
    }
}

float activationFunction(float x)
{
	return std::max(0.0f, x);
}

float activationFunctionDerivative(float x)
{
	return x > 0.0f ? 1.0f : 0.0f;
}
// you can use your own code here!
Eigen::RowVectorXf softmax(const Eigen::RowVectorXf& x)
{
    Eigen::RowVectorXf exp_values = x.array().exp();
    return exp_values / exp_values.sum();
}

void NeuralNetwork::forward_prop(Eigen::RowVectorXf& input)
{
    layers.front()->block(0, 0, 1, layers.front()->size()-1) = input;
    for(unsigned int i = 1; i < neuron_layer_num.size(); i++)
    {
        (*layers[i]) = (*layers[i-1]) * (*weights[i-1]);
        layers[i]->block(0, 0, 1, neuron_layer_num[i]).unaryExpr(std::ptr_fun(activationFunction));
    }
    (*layers.back()) = softmax(*layers.back());
}

void NeuralNetwork::eval_err(Eigen::RowVectorXf& output)
{
	// calculate the errors made by neurons of last layer
	(*deltas.back()) = output - (*layers.back());

	// error calculation of hidden layers is different
	// we will begin by the last hidden layer
	// and we will continue till the first hidden layer
	for (unsigned int i = neuron_layer_num.size() - 2; i > 0; i--)
    {
		(*deltas[i]) = (*deltas[i + 1]) * (weights[i]->transpose());
	}
}

void NeuralNetwork::update_weights()
{
	// topology.size()-1 = weights.size()
	for (unsigned int i = 0; i < neuron_layer_num.size() - 1; i++)
	{
		// in this loop we are iterating over the different layers (from first hidden to output layer)
		// if this layer is the output layer, there is no bias neuron there, number of neurons specified = number of cols
		// if this layer not the output layer, there is a bias neuron and number of neurons specified = number of cols -1
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

Eigen::MatrixXf softmax_derivative(const Eigen::RowVectorXf& x)
{
    Eigen::MatrixXf jacobian(x.size(), x.size());
    for (int i = 0; i < x.size(); i++) {
        for (int j = 0; j < x.size(); j++) {
            if (i == j) {
                jacobian(i, j) = x(i) * (1 - x(i));
            } else {
                jacobian(i, j) = -x(i) * x(j);
            }
        }
    }
    return jacobian;
}

void NeuralNetwork::backward_prop(Eigen::RowVectorXf& output)
{
    // Change this later
    OneHot onehot(10);
    Eigen::RowVectorXf output_encoded = onehot.encode(static_cast<int>(output(0)));
    eval_err(output_encoded);
    (*deltas.back()) = (*deltas.back()) * softmax_derivative(*layers.back());
	update_weights();
}

void NeuralNetwork::train(std::vector<Eigen::RowVectorXf*> input_data, std::vector<Eigen::RowVectorXf*> output_data)
{
	for (unsigned int i = 0; i < input_data.size(); i++)
	{
		std::cout << "Input to neural network is : " << *input_data[i] << std::endl;
		forward_prop(*input_data[i]);
		std::cout << "Expected output is : " << *output_data[i] << std::endl;
		std::cout << "Output produced is : " << *layers.back() << std::endl;
		backward_prop(*output_data[i]);
		std::cout << "MSE : " << std::sqrt((*deltas.back()).dot((*deltas.back())) / deltas.back()->size()) << std::endl;
	}
}

void ReadCSV(std::string filename, std::vector<Eigen::RowVectorXf*>& data)
{
	data.clear();
	std::ifstream file(filename);
	std::string line, word;
	// determine number of columns in file
	getline(file, line, '\n');
	std::stringstream ss(line);
	std::vector<float> parsed_vec;
	while (getline(ss, word, ','))
	{
		parsed_vec.push_back(float(std::stof(&word[0])));
	}
	unsigned int cols = parsed_vec.size();
	data.push_back(new Eigen::RowVectorXf(cols));
	for (unsigned int i = 0; i < cols; i++)
	{
		data.back()->coeffRef(1, i) = parsed_vec[i];
	}

	// read the file
	if (file.is_open())
	{
		while (getline(file, line, '\n'))
		{
			std::stringstream ss(line);
			data.push_back(new Eigen::RowVectorXf(1, cols));
			unsigned int i = 0;
			while (getline(ss, word, ','))
			{
				data.back()->coeffRef(i) = float(std::stof(&word[0]));
				i++;
			}
		}
	}
}

void genData(std::string filename)
{
	std::ofstream file1(filename + "-in");
	std::ofstream file2(filename + "-out");
	for (unsigned int r = 0; r < 1000; r++) {
		float x = rand() / float(RAND_MAX);
		float y = rand() / float(RAND_MAX);
		file1 << x << ',' << y << std::endl;
		file2 << 2 * x + 10 + y << std::endl;
	}
	file1.close();
	file2.close();
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
    n.train(mnist_train_vectors, mnist_train_label_vectors);

	for(int i; i < n.neuron_layer_num.size(); i++)
	{
		std::cout << *n.weights[i] << std::endl;
	}
	
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