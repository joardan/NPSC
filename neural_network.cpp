#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>


class OneHot {
private:
    unsigned char num_classes;

public:
    // Constructor
    OneHot(unsigned char classes) : num_classes(classes) {}

    // One-hot encoder function
    Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> encode(int label) const
    {
        Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> encoded(num_classes);
        encoded.setZero();
        encoded(label) = 1;
        return encoded;
    }

    // One-hot decoder function
    unsigned char decode(const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>& encoded) const
    {
        for (int i = 0; i < encoded.size(); ++i)
        {
            if (encoded(i) == 1)
            {
                return i;
            }
        }
        return -1; // Invalid one-hot encoding
    }
};

Eigen::MatrixXf relu(const Eigen::MatrixXf& input)
{
    return input.array().max(0.0);
}

Eigen::MatrixXf feed_forward(const Eigen::MatrixXf& input, const Eigen::MatrixXf& weight) {
    // Create a bias matrix by concatenating a column vector of 1s to the input matrix
    Eigen::MatrixXf input_with_bias(input.rows(), input.cols() + 1);
    input_with_bias << input, Eigen::MatrixXf::Ones(input.rows(), 1);

    // Calculate the net matrix
    Eigen::MatrixXf net = input_with_bias * weight;

    // Calculate the output matrix by applying the activation function element-wise to the net matrix
    Eigen::MatrixXf output = relu(net);

    return output;
}

Eigen::MatrixXf init_weight(int row, int col)
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
    Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> classes = encoderDecoder.encode(int(maxIndex2));
    std::cout << classes << std::endl << std::endl << std::endl;


    return error;
}

Eigen::MatrixXf backprop()
{}

int main()
{
    Eigen::MatrixXf input(1, 10);
    input << 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5;
    Eigen::MatrixXf output(1, 5);
    output << 0, 0, 0, 1, 0;
    Eigen::MatrixXf class_output(1, 5);
    class_output << 0, 0, 0, 1, 0;
    Eigen::MatrixXf weight = init_weight(11, 5);
    double error = evaluate_err(input, weight, output, class_output);
    std::cout << error;
    Eigen::MatrixXf result = feed_forward(input, weight);
    std::cout << weight << std::endl;
    std::cout << input << std::endl;
    std::cout << result << std::endl;
}