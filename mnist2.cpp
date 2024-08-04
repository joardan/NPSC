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

// NOT USED BECAUSE THE MAGIC NUMBER IS NOT NEEDED, BUT MIGHT BE USEFUL LATER
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

std::vector<Eigen::RowVectorXd*> mnistImageToEigenVector(unsigned char** mnist_image, int num_items, int image_size)
{
    std::vector<Eigen::RowVectorXd*> vectors;

    for (int i = 0; i < num_items; ++i) {
        Eigen::RowVectorXd* rowVector = new Eigen::RowVectorXd(image_size);
        for (int j = 0; j < image_size; ++j) {
            (*rowVector)(j) = static_cast<double>(mnist_image[i][j]) / 255.0f;
        }
        vectors.push_back(rowVector);
    }

    return vectors;
}

std::vector<Eigen::RowVectorXd*> mnistLabelToEigenVector(unsigned char* mnist_label, int num_items)
{
    std::vector<Eigen::RowVectorXd*> vectors;

    for (int i = 0; i < num_items; ++i) {
        Eigen::RowVectorXd* rowVector = new Eigen::RowVectorXd(1);
        (*rowVector)(0) = static_cast<double>(mnist_label[i]);
        vectors.push_back(rowVector);
    }

    return vectors;
}

// TEST MAIN FILE, MADE JUST TO TEST READING MNIST DATASET ITSELF.
/*
int main() {
    unsigned char *mnist_label_test = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-labels.idx1-ubyte", 10000);
    unsigned char **mnist_image_test = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/t10k-images.idx3-ubyte", 10000, 784);
    unsigned char *mnist_label_train = read_mnist_label("/media/joardan/Harddisk/Project/NPSC/dataset/train-labels.idx1-ubyte", 60000);
    unsigned char **mnist_image_train = read_mnist_image("/media/joardan/Harddisk/Project/NPSC/dataset/train-images.idx3-ubyte", 60000, 784);

    // Convert MNIST images to Eigen vectors
    std::vector<Eigen::RowVectorXd*> mnist_train_vectors = mnistImageToEigenVector(mnist_image_train, 60000, 784);
    std::vector<Eigen::RowVectorXd*> mnist_train_label_vectors = mnistLabelToEigenVector(mnist_label_train, 60000);

    // Example of accessing a specific image (vector) in the matrix
    Eigen::RowVectorXd* example_image_vector = mnist_train_vectors[111];

    // Displaying the example image vector
    std::cout << "Example image vector: " << std::endl;
    std::cout << *example_image_vector << std::endl;

    // Displaying the example image using OpenCV
    displayMNISTImage(mnist_image_train[111], 28, 28);

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
*/