#include <iostream>
#include <vector>
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
    image_processor(const std::string& file_path)
    {
        image = cv::imread(file_path, cv::IMREAD_GRAYSCALE);
        if (!image.data)
        {
            throw std::runtime_error("No image data");
        }
    }

    // Threshold the image
    void threshold_image()
    {
        cv::GaussianBlur(image, image, cv::Size(5,5), 0);
        cv::threshold(image,image,0,255,cv::THRESH_BINARY+cv::THRESH_OTSU);
        cv::threshold(image, image, 127, 255, cv::THRESH_BINARY_INV);
    }

    // Find contours in the thresholded image
    void find_contours()
    {
        cv::findContours(image, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    }

    // Find indices of likely number contours based on area
    void find_number_indices()
    {
        double avgArea = 0;
        double sumArea = 0;
        for (const auto& contour : contours)
        {
            sumArea += cv::contourArea(contour);
        }
        avgArea = sumArea / (5 * contours.size());

        for (size_t i = 0; i < contours.size(); ++i)
        {
            double area = cv::contourArea(contours[i]);
            if (area > avgArea)
            {
                idx.push_back(i);
            }
        }
    }

    // Create bounding boxes for the likely number areas
    void create_bounding_boxes()
    {
        if (idx.empty()) return;
        int margin = 10;

        for (int i : idx)
        {
            cv::Rect rect = cv::boundingRect(contours[i]);
            int difference = 0;

            // Adjust the bounding rectangle to square and add margin
            if (rect.width > rect.height)
            {
                difference = rect.width - rect.height;
                rect.x -= margin;
                rect.width += 2 * margin;
                rect.height = rect.width;
                rect.y -= margin + (difference / 2);
            }
            else
            {
                difference = rect.height - rect.width;
                rect.y -= margin;
                rect.height += 2 * margin;
                rect.width = rect.height;
                rect.x -= margin + (difference / 2);
            }
            rect.x = std::max(0, rect.x);
            rect.y = std::max(0, rect.y);
            rect.width = std::min(rect.width, image.cols - rect.x);
            rect.height = std::min(rect.height, image.rows - rect.y);

            boundingRects.push_back(rect);
        }

        // First, sort the bounding boxes by their y-values (top to bottom)
        std::sort(boundingRects.begin(), boundingRects.end(), [](const cv::Rect& a, const cv::Rect& b)
        {
            // If the y-values are close, sort by x-values (left to right)
            if (std::abs(a.y - b.y) < 20)  // Tolerance to group them into the same row
            {
                return a.x < b.x;
            }
            return a.y < b.y;
        });
    }

    // Crop images and preprocess (resize, dilate, blur)
    void crop_and_preprocess()
    {
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Size targetSize(28, 28);

        for (const auto& rect : boundingRects)
        {
            cv::Mat img = image(rect).clone();
            cv::resize(img, img, targetSize);
            processedImages.push_back(img);
        }
    }

    // Convert processed images to Eigen matrices
    std::vector<std::vector<Eigen::MatrixXd*>> convert_to_eigen()
    {
        std::vector<std::vector<Eigen::MatrixXd*>> image_matrices;
        for(int i = 0; i < processedImages.size(); i++)
        {
            Eigen::MatrixXd* dest = new Eigen::MatrixXd(28, 28);
            cv::cv2eigen(processedImages[i], *dest);
            *dest = *dest / 255.0f;
            image_matrices.push_back({(dest)});
        }
        return image_matrices;
    }

    // Optional display method to visualize steps (for debugging)
    void display_image(const std::string& window_name, const cv::Mat& img)
    {
        cv::namedWindow(window_name, cv::WINDOW_KEEPRATIO);
        cv::imshow(window_name, img);
        cv::waitKey(0);
    }

    void process()
    {
        threshold_image();
        find_contours();
        find_number_indices();
        create_bounding_boxes();
        crop_and_preprocess();
    }

    void process_demo()
    {
        display_image("Image", image);
        threshold_image();
        display_image("Image", image);
        find_contours();
        find_number_indices();
        create_bounding_boxes();
        cv::Scalar intensity(100);
        for(auto rect : boundingRects)
        {
            cv::Mat img = image.clone();
            cv::rectangle(img, rect, intensity, 2);
            display_image("Image", img);
        }
        for (size_t i = 0; i < boundingRects.size(); ++i)
        {
            cv::Mat croppedImg = image(boundingRects[i]).clone();
            std::string window_name = "Boxed Image " + std::to_string(i + 1);
            display_image(window_name, croppedImg);
        }
        crop_and_preprocess();
        for (const auto& img : processedImages)
        {
            display_image("Processed Image", img);
        }
        cv::destroyWindow("Processed Image");
    }

    void process_demo_lite()
    {
        display_image("Image", image);
        threshold_image();
        display_image("Image", image);
        find_contours();
        find_number_indices();
        create_bounding_boxes();
        cv::Scalar intensity(100);
        for(auto rect : boundingRects)
        {
            cv::Mat img = image.clone();
            cv::rectangle(img, rect, intensity, 2);
            display_image("Image", img);
        }
        for (size_t i = 0; i < boundingRects.size(); ++i)
        {
            cv::Mat croppedImg = image(boundingRects[i]).clone();
            display_image("Image", croppedImg);
        }
        crop_and_preprocess();
        for (const auto& img : processedImages)
        {
            display_image("Image", img);
        }
        cv::destroyAllWindows();
    }
};

/*
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cout << "Usage: Image_processing.out <Image_Path>" << std::endl;
        return -1;
    }

    try
    {
        // Create image processor object
        image_processor processor(argv[1]);

        // Run the image processing pipeline
        processor.process_demo();

        // Convert processed images to Eigen matrices
        std::vector<std::vector<Eigen::MatrixXd*>> eigen_matrices = processor.convert_to_eigen();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return -1;
    }

    return 0;
}
*/