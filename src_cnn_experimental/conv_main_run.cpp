#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <Eigen/Dense>
#include "mnist2.hpp"
#include "encoder.hpp"
#include "multi_numbers_processing.hpp"
#include "conv_layer_unoptimised.hpp"
#include "conv_max_pool.hpp"
#include "conv_fc_layer.hpp"
#include "conv_function.hpp"

int main(int argc, char* argv[])
{
    // Initialize layers
    ConvolutionalLayer conv_layer(28, 28, 1, 5, 42);
    MaxPoolingLayer pool_layer(24, 24, 42, 2);
    FullyConnectedLayer fc_layer(pool_layer.output_depth * pool_layer.output_width * pool_layer.output_height, 10);
    std::vector<std::vector<Eigen::MatrixXd*>> image_matrices;

    load_model(conv_layer, fc_layer, "./trained_model_data5");
    bool running = true;
    bool preprocessed = false;

    if (argc != 2)
    {
        printf("usage: Image_processing.out <Image_Path>\n");
        return -1;
    }
    image_processor processor(argv[1]);

    while (running) {
        displayMenu();
        int choice;
        std::cin >> choice;

        switch (choice) {
            case 1:
            {
                if (preprocessed == false)
                {
                    processor.process_demo();
                    image_matrices = processor.convert_to_eigen();
                    preprocessed = true;
                }
                break;
            }
            case 2:
            {
                if (preprocessed == false)
                {
                    processor.process_demo_lite();
                    image_matrices = processor.convert_to_eigen();
                    preprocessed = true;
                }
                break;
            }
            case 3:
            {
                if (preprocessed == false)
                {
                    processor.process();
                    image_matrices = processor.convert_to_eigen();
                    preprocessed = true;
                }
                break;
            }
            case 4:
            {
                if(image_matrices.empty())
                {
                    std::cout << "No image found" << std::endl;
                    break;
                }
                else if(preprocessed == false)
                {
                    std::cout << "Process the image first" << std::endl;
                }
                else
                {
                    predict(conv_layer, pool_layer, fc_layer, image_matrices);
                    break;
                }
            }
            case 5:
                running = false;
                std::cout << "Exitted\n";
                break;
            default:
                std::cout << "Invalid choice. Please try again\n";
        }
    }

    for (auto& batch : image_matrices)
    {
        for (auto& matrix : batch)
        {
            delete matrix;
        }
    }   

    return 0;
}

// trained_model_data is for 3x3 filter 32 filter_num
// trained_model_data2-5 is for 5x5 filter with 42 filter_num