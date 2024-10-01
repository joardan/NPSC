#include <random>
#include "initialiser.hpp"

void Initialiser::init(Eigen::MatrixXd& weights, const std::string& initialiser)
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

void Initialiser::initialiseHe(Eigen::MatrixXd& weights)
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

void Initialiser::initialiseXavier(Eigen::MatrixXd& weights)
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