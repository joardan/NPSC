#ifndef LAYER_HPP
#define LAYER_HPP

#include <Eigen/Dense>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/eigen.hpp>


class image_processor
{
private:
    cv::Mat image;  // Store the original image
    std::vector<std::vector<cv::Point>> contours;
    std::vector<int> idx;
    std::vector<cv::Rect> boundingRects;
    std::vector<cv::Mat> processedImages;

public:
    // Constructor to load the image
    image_processor(const std::string& file_path);

    // Threshold the image
    void threshold_image();

    // Find contours in the thresholded image
    void find_contours();

    // Find indices of likely number contours based on area
    void find_number_indices();

    // Create bounding boxes for the likely number areas
    void create_bounding_boxes();

    // Crop images and preprocess (resize, dilate, blur)
    void crop_and_preprocess();

    // Convert processed images to Eigen matrices
    std::vector<std::vector<Eigen::MatrixXd*>> convert_to_eigen();

    // Optional display method to visualize steps (for debugging)
    void display_image(const std::string& window_name, const cv::Mat& img);

    void process();
    void process_demo();
    void process_demo_lite();
};

#endif