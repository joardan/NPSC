#include <iostream>
#include "encoder.hpp"

// Constructor definition
OneHot::OneHot(unsigned char classes) : num_classes(classes) {}

// One-hot encoder function definition
Eigen::Matrix<unsigned char, 1, Eigen::Dynamic> OneHot::encode(int label) const
{
    Eigen::Matrix<unsigned char, 1, Eigen::Dynamic> encoded(num_classes);
    encoded.setZero();
    encoded(label) = 1;
    return encoded;
}

// One-hot decoder function definition
unsigned char OneHot::decode(const Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>& encoded) const
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


// TEST MAIN FILE, UNCOMMENT AND RUN CPP FILE TO TEST.
/*
int main()
{
    const unsigned char numClasses = 10; // MNIST has 10 classes (digits 0-9)
    unsigned char label = 9; // Example label

    // Create an instance of the encoder/decoder
    OneHot encoderDecoder(numClasses);

    // Encode label
    Eigen::Matrix<unsigned char, Eigen::Dynamic, 1> encoded = encoderDecoder.encode(label);

    // Decode one-hot encoding
    int decodedLabel = encoderDecoder.decode(encoded);

    // Output original label and decoded label
    std::cout << "Original Label: " << static_cast<int>(label) << std::endl;
    std::cout << encoded << std::endl;
    std::cout << "Decoded Label: " << decodedLabel << std::endl;

    return 0;
}
*/
