#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

int main(int argc, char** argv )
{
    // Make sure there's input image when running the program
    if (argc != 2)
    {
        printf("usage: Image_processing.out <Image_Path>\n");
        return -1;
    }
    
    // Read image in grayscale
    cv::Mat image;
    image = imread( argv[1], cv::IMREAD_GRAYSCALE );

    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }


    // .........................................................Delete later
    cv::namedWindow("Display Image2", cv::WINDOW_KEEPRATIO );
    cv::imshow("Display Image2", image);
    cv::waitKey(0);
    // .........................................................

    // Use threshold to clarify image contrast
    cv::Mat thresh;
    cv::threshold(image, thresh, 127, 255, cv::THRESH_BINARY_INV);

    // Find contour to narrow down the image
    std::vector<std::vector<cv::Point>> contour;
    cv::findContours(thresh, contour, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find big areas using average where it's likely to be a number
    std::vector<int> idx;
    double avgArea = 0;
    double sumArea = 0;
    for (size_t i = 0; i < contour.size(); ++i)
    {
        sumArea += cv::contourArea(contour[i])
    }
    avgArea = sumArea / contour.size()
    for (i = 0; i < contour.size(); ++i)
    {
        double area = cv::contourArea(contour[i]);
        if (area > avgArea)
        {
            idx.push_back(area);
        }
    }

    

    // Create a rectangle boundary
    std::vector<cv::Rect> boundingRects;
    if (idx.size() > 0)
    {
        for (size_t i = 0; i < contour.size(); ++i)
        {
            boundingRects.push_back(cv::boundingRect(contour[maxAreaIdx]));
            cv::Scalar intensity(100);

            // Set Margin of the boundary to crop the image, done to make sure there's white space around the number just like MNIST
            int margin = 10;

            // Expand the bounding rectangle by adding the margin, then bound it in a square before recentre again
            if (boundingRects[i].width > boundingRects[i].height)
            {
                boundingRects[i].y -= 2 * margin;
                boundingRects[i].x -= margin;
                boundingRects[i].width += 2 * margin;
                boundingRects[i].height = boundingRects[i].width;
            }
            else
            {
                boundingRects[i].x -= 2 * margin;
                boundingRects[i].y -= margin;
                boundingRects[i].height += 2 * margin;
                boundingRects[i].width = boundingRects[i].height;
            }

            // Draw rectangle
            cv::rectangle(thresh, boundingRects[i], intensity, 1);
        }
    }


    // .........................................................Delete later
    cv::namedWindow("Display Image1", cv::WINDOW_KEEPRATIO );
    cv::imshow("Display Image1", thresh);
    cv::waitKey(0);
    // .........................................................



    // Crop the image using the boundingRect
    thresh = thresh(boundingRect).clone();


    // .........................................................Delete later
    cv::namedWindow("Display Image3", cv::WINDOW_KEEPRATIO );
    cv::imshow("Display Image3", thresh);
    cv::waitKey(0);
    // .........................................................


    // Remove Noise, Dilate, then Blur
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);
    cv::dilate(thresh, thresh, kernel);
    cv::GaussianBlur(thresh, thresh, cv::Size(5, 5), 0);

    // Resize to 28 x 28 to match the MNIST dataset
    cv::Size targetSize(28, 28);
    resize(thresh, thresh, targetSize);

    // Display Image
    cv::namedWindow("Display Image", cv::WINDOW_KEEPRATIO );
    cv::imshow("Display Image", thresh);
    cv::waitKey(0);
    cv::destroyWindow("Display Image");
    return 0;
}