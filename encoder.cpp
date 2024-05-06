#include <iostream>
#include <vector>
#include <Eigen/Core>

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


int main()
{
    const int numClasses = 10; // MNIST has 10 classes (digits 0-9)
    int label = 9; // Example label

    // Create an instance of the encoder/decoder
    OneHot encoderDecoder(numClasses);

    // Encode label
    Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> encoded = encoderDecoder.encode(label);

    // Decode one-hot encoding
    int decodedLabel = encoderDecoder.decode(encoded);

    // Output original label and decoded label
    std::cout << "Original Label: " << label << std::endl;
    std::cout << encoded << std::endl;
    std::cout << "Decoded Label: " << decodedLabel << std::endl;

    return 0;
}
