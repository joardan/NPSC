#ifndef TESTER_HPP
#define TESTER_HPP

#include <Eigen/Dense>
#include "neural_network.hpp"

class NeuralNetworkTester
{
    public:
        double calculate_accuracy(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, NeuralNetwork& model);
        double kFoldCrossValidation(const std::vector<Eigen::RowVectorXd*>& inputs, const std::vector<Eigen::RowVectorXd*>& targets, int k, int epochs, int early_stopping_patience);

    private:
        int max_arg(const Eigen::RowVectorXd& x);
};

#endif