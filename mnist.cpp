#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>


void displayMNISTImage(unsigned char* image, int rows, int cols)
{
    // Create an OpenCV Mat object to hold the image data
    cv::Mat img(rows, cols, CV_8U, image);

    // Display the image using OpenCV
    cv::namedWindow("MNIST Image", cv::WINDOW_KEEPRATIO);
    cv::imshow("MNIST Image", img);
    cv::waitKey(5000); // Wait indefinitely until a key is pressed
    cv::destroyAllWindows();
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

unsigned char* read_mnist_label(const std::string& file_path, int num_items) {
    std::ifstream file(file_path, std::ios::binary);
    if (file.is_open()) {
        // Skip the first two bytes
        file.seekg(2 * sizeof(int), std::ios::beg);

        unsigned char *mnist_label = new unsigned char[num_items];
        for (int i = 0; i < num_items; ++i) {
            file.read((char *)&mnist_label[i], sizeof(unsigned char));
        }
        file.close();
        return mnist_label;
    }
    else
    {
        std::cerr << "Can't open file: " << file_path << std::endl;
        return nullptr;
    }
}

unsigned char** read_mnist_image(const std::string& file_path, int num_items, int image_size)
{
    std::ifstream file(file_path, std::ios::binary);
    if (file.is_open()) {
        file.seekg(4 * sizeof(int), std::ios::beg);
        unsigned char **mnist_image = new unsigned char*[num_items];
        for(int i = 0; i < num_items; i++) {
            mnist_image[i] = new unsigned char[image_size];
            file.read((char *)mnist_image[i], image_size);
        }
        file.close();
        return mnist_image;
    }
    else
    {
        std::cerr << "can't open file";
        return nullptr;
    }
}

Eigen::MatrixXd mnistImageToEigenMatrix(unsigned char** mnist_image, int num_items, int image_size) {
    Eigen::MatrixXd matrix(image_size, num_items);

    for (int i = 0; i < num_items; ++i) {
        for (int j = 0; j < image_size; ++j) {
            matrix(j, i) = static_cast<double>(mnist_image[i][j]);
        }
    }

    return matrix;
}

int main() {
    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/train-images.idx3-ubyte", 60000, 784);
    Eigen::MatrixXd mnist_train_matrix = mnistImageToEigenMatrix(mnist_image_train, 60000, 784);

    // Example of accessing a specific image (column) in the matrix
    Eigen::VectorXd example_image = mnist_train_matrix.col(111);

    // Displaying the example image
    std::cout << "Example image: " << std::endl;
    std::cout << example_image.reshaped<Eigen::RowMajor>(28, 28);
    displayMNISTImage(mnist_image_train[111], 28, 28);
    delete[] mnist_label_test;
    delete[] mnist_label_train;
    delete[] mnist_image_test;
    delete[] mnist_image_train;
    return 0;
}