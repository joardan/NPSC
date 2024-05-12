#include "neural_network2.hpp"
#include <iostream>
#include <fstream>

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
	return tanhf(x);
}

float activationFunctionDerivative(float x)
{
	return 1 - tanhf(x) * tanhf(x);
}
// you can use your own code here!

void NeuralNetwork::forward_prop(Eigen::RowVectorXf& input)
{
    layers.front()->block(0, 0, 1, layers.front()->size()-1) = input;
    for(unsigned int i = 1; i < neuron_layer_num.size(); i++)
    {
        (*layers[i]) = (*layers[i-1]) * (*weights[i-1]);
        layers[i]->block(0, 0, 1, neuron_layer_num[i]).unaryExpr(std::ptr_fun(activationFunction));
    }
}

void NeuralNetwork::eval_err(Eigen::RowVectorXf& output)
{
	// calculate the errors made by neurons of last layer
	(*deltas.back()) = output - (*layers.back());

	// error calculation of hidden layers is different
	// we will begin by the last hidden layer
	// and we will continue till the first hidden layer
	for (unsigned int i = neuron_layer_num.size() - 2; i > 0; i--) {
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

void NeuralNetwork::backward_prop(Eigen::RowVectorXf& output)
{
	eval_err(output);
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

/*
int main()
{
    NeuralNetwork n({2, 3, 1}, 0.005);
    std::vector<Eigen::RowVectorXf*> in_dat, out_dat;
    genData("test");
    ReadCSV("test-in", in_dat);
    ReadCSV("test-out", out_dat);
    n.train(in_dat, out_dat);
    return 0;
}
*/