#include <vector>
#include <random>
#include <iostream>
#include "tester.hpp"

// Calculate accuracy from test dataset
double NeuralNetworkTester::calculate_accuracy(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, NeuralNetwork& model)
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

// Kfold to get accuracy without overfitting and without using test dataset
double NeuralNetworkTester::kFoldCrossValidation(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, int k, int epochs, int early_stopping_patience)
{
    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::random_shuffle(indices.begin(), indices.end());

    double total_accuracy = 0.0;
    size_t fold_size = inputs.size() / k;

    for (int i = 0; i < k; ++i)
    {
        NeuralNetwork model(0.0242);
        model.add_Layer(784, 64);
        model.add_Layer(64, 32);
        model.add_Layer(32, 16);
        model.add_Layer(16, 10);

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

        int best_epoch = 0;
        double best_accuracy = 0.0;
        int patience_counter = 0;

        for (int epoch = 1; epoch <= epochs; ++epoch)
        {
            model.train(train_inputs, train_targets);

            double test_accuracy = calculate_accuracy(test_inputs, test_targets, model);
            std::cout << "Fold " << i + 1 << " - Epoch " << epoch << " - Test Accuracy: " << test_accuracy << std::endl;

            if (test_accuracy > best_accuracy)
            {
                best_accuracy = test_accuracy;
                best_epoch = epoch;
                patience_counter = 0;
            }
            else
            {
                patience_counter++;
            }

            if (patience_counter >= early_stopping_patience)
            {
                std::cout << "Early stopping at epoch " << epoch << " with best test accuracy of " << best_accuracy << " at epoch " << best_epoch << "." << std::endl;
                break;
            }
        }

        total_accuracy += best_accuracy;
    }

    return total_accuracy / k;
}

// Get max arg to get the predicted output from probability output
int NeuralNetworkTester::max_arg(const Eigen::RowVectorXd& x)
{
    Eigen::Index arg;
    x.maxCoeff(&arg);
    return arg;
}
