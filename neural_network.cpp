#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <tuple>
#include <algorithm>
#include "mnist.hpp"
#include "encoder.hpp"
#include "functions.hpp"

#define MAX_SAMPLES 100 // Maximum number of samples for training
#define MAX_ITERATIONS 500
#define LEARNING_RATE 0.1


Eigen::MatrixXf feed_forward(const Eigen::MatrixXf& input, const Eigen::MatrixXf& weight, const Eigen::MatrixXf& bias)
{
    // Create a bias matrix by concatenating a column vector of 1s to the input matrix
    Eigen::MatrixXf input_with_bias(input.rows(), input.cols() + 1);
    input_with_bias << input, bias;

    // Calculate the net matrix
    Eigen::MatrixXf net = input_with_bias * weight;

    // Calculate the output matrix by applying the activation function element-wise to the net matrix
    Eigen::MatrixXf output = relu(net);

    return output;
}

Eigen::MatrixXf init_params(int row, int col)
{
    srand(1);
    Eigen::MatrixXf weight = Eigen::MatrixXf::Random(row, col);
    return weight;
}

double evaluate_err(const Eigen::MatrixXf& input, const Eigen::MatrixXf& weight, 
Eigen::MatrixXf& target_output, Eigen::MatrixXf& target_class)
{
    Eigen::MatrixXf output = feed_forward(input, weight);
    int sample_count = output.rows();
    int output_count = output.cols();
    std::cout << (target_output - output).array().square() << std::endl;
    std::cout << (target_output - output).array().square().sum() << std::endl;
    std::cout << sample_count << std::endl << output_count << std::endl;
    double error = (target_output - output).array().square().sum() / (sample_count * output_count);

    Eigen::Index maxIndex, maxIndex2;
    std::cout << output.colwise().sum();
    float maxNorm = output.maxCoeff(&maxIndex, &maxIndex2);
    std::cout << maxNorm << std::endl;
    std::cout << maxIndex2 << std::endl;
    // Calculate the classification error

    OneHot encoderDecoder(output_count);
    // Encode label
    Eigen::Matrix<unsigned char, 1, Eigen::Dynamic> classes = encoderDecoder.encode(int(maxIndex2));
    std::cout << classes << std::endl << std::endl << std::endl;
    std::cout << target_class;
    int num_mismatches = (classes.array() != target_class.cast<unsigned char>().array()).sum();
    double classification_error = static_cast<double>(num_mismatches) / sample_count;
    std::cout << std::endl << std::endl << std::endl << classification_error << std::endl << std::endl;
    return error;
}

Eigen::MatrixXf backprop(const Eigen::MatrixXf& input, const Eigen::MatrixXf& weight, 
                         const Eigen::MatrixXf& target_output, const Eigen::MatrixXf& target_class,
                         double learning_rate)
{
    // Forward Pass
    Eigen::MatrixXf output = feed_forward(input, weight);

    // Check dimensions
    std::cout << "Input dimensions: " << input.rows() << "x" << input.cols() << std::endl;
    std::cout << "Weight dimensions: " << weight.rows() << "x" << weight.cols() << std::endl;
    
    // Compute Loss
    Eigen::MatrixXf loss_grad = -(target_output - output);

    // Check dimensions
    std::cout << "Loss gradient dimensions: " << loss_grad.rows() << "x" << loss_grad.cols() << std::endl;

    Eigen::MatrixXf input_with_bias(input.rows(), input.cols() + 1);
    input_with_bias << input, Eigen::MatrixXf::Ones(input.rows(), 1);
    
    // Check dimensions
    std::cout << "Input with bias dimensions: " << input_with_bias.rows() << "x" << input_with_bias.cols() << std::endl;
    
    // Calculate the net matrix
    Eigen::MatrixXf net = input_with_bias * weight;
    
    // Check dimensions
    std::cout << "Net matrix dimensions: " << net.rows() << "x" << net.cols() << std::endl;

    Eigen::MatrixXf delta = loss_grad.array() * (net.array() > 0).cast<float>();

    // Check dimensions
    std::cout << "Delta dimensions: " << delta.rows() << "x" << delta.cols() << std::endl;

    // Calculate weights delta
    Eigen::MatrixXf weights_delta = learning_rate * input_with_bias.transpose() * delta;

    // Check dimensions
    std::cout << "Weights delta dimensions: " << weights_delta.rows() << "x" << weights_delta.cols() << std::endl;

    // Update Weights
    Eigen::MatrixXf updated_weight = weight - weights_delta;

    return updated_weight;
}


std::tuple<Eigen::MatrixXf, std::vector<double>, std::vector<double>, std::vector<double>,
           std::vector<double>, std::vector<double>, std::vector<double>> 
train(const Eigen::MatrixXf& training_set, Eigen::MatrixXf& validation_set, Eigen::MatrixXf& test_set) {
    int num_inputs = 784;
    int num_outputs = 10; // Assuming output size is the same as input size

    // Initialize weights and biases
    Eigen::MatrixXf weights = init_weight(num_inputs + 1, num_outputs); // +1 for bias
    Eigen::MatrixXf bias_training = Eigen::MatrixXf::Ones(training_set.rows(), 1);
    Eigen::MatrixXf bias_validate = Eigen::MatrixXf::Ones(validation_set.rows(), 1);
    Eigen::MatrixXf bias_test = Eigen::MatrixXf::Ones(test_set.rows(), 1);

    // Arrays to store errors
    std::vector<double> error_train;
    std::vector<double> classification_error_train;
    std::vector<double> error_validate;
    std::vector<double> classification_error_validate;
    std::vector<double> error_test;
    std::vector<double> classification_error_test;

    int epoch = 0;
    while (epoch < MAX_ITERATIONS) {
        // Limit the number of samples used for training
        Eigen::MatrixXf truncated_training_set = training_set.topRows(std::min(static_cast<int>(training_set.rows()), MAX_SAMPLES));

        // Backpropagation
        weights = backprop(truncated_training_set, weights, truncated_training_set, truncated_training_set, LEARNING_RATE);

        // Calculate errors
        double error_train_epoch = evaluate_err(truncated_training_set, weights, truncated_training_set, truncated_training_set);
        error_train.push_back(error_train_epoch);
        classification_error_train.push_back(0.0); // Placeholder, as classification error is not yet implemented
        double error_validate_epoch = evaluate_err(validation_set, weights, validation_set, validation_set);
        error_validate.push_back(error_validate_epoch);
        classification_error_validate.push_back(0.0); // Placeholder, as classification error is not yet implemented
        double error_test_epoch = evaluate_err(test_set, weights, test_set, test_set);
        error_test.push_back(error_test_epoch);
        classification_error_test.push_back(0.0); // Placeholder, as classification error is not yet implemented

        epoch++;
    }

    return std::make_tuple(weights, error_train, classification_error_train,
                           error_validate, classification_error_validate,
                           error_test, classification_error_test);
}

int main()
{
    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/train-images.idx3-ubyte", 60000, 784);
    Eigen::MatrixXf mnist_train_matrix = mnistImageToEigenMatrix(mnist_image_train, 60000, 784);
    Eigen::MatrixXf mnist_test_matrix = mnistImageToEigenMatrix(mnist_image_train, 10000, 784);
    Eigen::MatrixXf::Ones bias(input.rows(), 1);

    // Example of accessing a specific image (column) in the matrix
    Eigen::VectorXf example_image = mnist_train_matrix.row(111);

    // Displaying the example image
    std::cout << "Example image: " << std::endl;
    std::cout << example_image.reshaped<Eigen::RowMajor>(28, 28);
    displayMNISTImage(mnist_image_train[111], 28, 28);
    
    Eigen::MatrixXf input(1, 10);
    input << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;

    Eigen::MatrixXf output(1, 5);
    output << 0, 0, 0, 1, 0;

    Eigen::MatrixXf class_output(1, 5);
    class_output << 0, 0, 0, 1, 0;

    Eigen::MatrixXf weight = init_weight(11, 5); // Assuming weights are initialized

    // Evaluate error
    double error = evaluate_err(input, weight, output, class_output);
    std::cout << "Error: " << error << std::endl;

    // Perform feed-forward
    Eigen::MatrixXf result = feed_forward(input, weight);
    std::cout << "Result:" << std::endl << result << std::endl;

    // Training set, validation set, and test set initialization (to be loaded from files)
    Eigen::MatrixXf training_set, validation_set, test_set;

    // Call the train function
    auto result_tuple = train(training_set, validation_set, test_set);

    // Retrieve results from the tuple
    Eigen::MatrixXf trained_weights = std::get<0>(result_tuple);
    std::vector<double> error_train = std::get<1>(result_tuple);
    std::vector<double> classification_error_train = std::get<2>(result_tuple);
    std::vector<double> error_validate = std::get<3>(result_tuple);
    std::vector<double> classification_error_validate = std::get<4>(result_tuple);
    std::vector<double> error_test = std::get<5>(result_tuple);
    std::vector<double> classification_error_test = std::get<6>(result_tuple);

    // Display or plot errors and classification errors
    delete[] mnist_label_test;
    delete[] mnist_label_train;
    delete[] mnist_image_test;
    delete[] mnist_image_train;

    return 0;
}