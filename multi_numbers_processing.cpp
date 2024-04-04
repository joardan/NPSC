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
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    //_ Find big areas that are larger than average where it's likely to be a number_____
    std::vector<int> idx;
    double avgArea = 0;
    double sumArea = 0;
    for (const auto& contour : contours)
    {
        sumArea += cv::contourArea(contour);
    }

    avgArea = sumArea / (1.5*contours.size());

    // Defined contourSize to optomise loop
    auto contourSize {contours.size()};
    for (auto i {0}; i < contourSize; ++i)
    {
        double area = cv::contourArea(contours[i]);
        if (area > avgArea)
        {
            idx.push_back(i);
        }
    }
    //___________________________________________________________________________________

    // Create a rectangle boundary
    std::vector<cv::Rect> boundingRects;
    if (idx.size() > 0)
    {
        cv::Scalar intensity(100);
        // Set Margin of the boundary to crop the image, done to make sure there's white space around the number just like MNIST
        auto margin = 10;
        auto difference = 0;
        for (auto i = 0; i < idx.size(); ++i)
        {
            boundingRects.push_back(cv::boundingRect(contours[idx[i]]));

            // Expand the bounding rectangle by adding the margin, then bound it in a square before recentre again
            if (boundingRects[i].width > boundingRects[i].height)
            {
                difference = boundingRects[i].width - boundingRects[i].height;
                boundingRects[i].x -= margin;
                boundingRects[i].width += 2 * margin;
                boundingRects[i].height = boundingRects[i].width;
                boundingRects[i].y -= margin + (difference / 2);
            }
            else
            {
                difference = boundingRects[i].height - boundingRects[i].width;
                boundingRects[i].y -= margin;
                boundingRects[i].height += 2 * margin;
                boundingRects[i].width = boundingRects[i].height;
                boundingRects[i].x -= margin + (difference / 2);
            }

            // Draw rectangle
            cv::rectangle(thresh, boundingRects[i], intensity, 1);

            // .........................................................Delete later
            cv::namedWindow("Display Image1", cv::WINDOW_KEEPRATIO );
            cv::imshow("Display Image1", thresh);
            cv::waitKey(0);
            // .........................................................
        }
    }


    // Crop the image using the boundingRect
    std::vector<cv::Mat> imgs;
    for (size_t i = 0; i < idx.size(); ++i)
    {
        imgs.push_back(thresh(boundingRects[i]).clone());
    }


    // .........................................................Delete later
    for (size_t i = 0; i < idx.size(); ++i)
    {
        cv::namedWindow("Display Image3", cv::WINDOW_KEEPRATIO );
        cv::imshow("Display Image3", imgs[i]);
        cv::waitKey(0);
    }
    // .........................................................

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::Size targetSize(28, 28);
    // Remove Noise, Dilate, then Blur
    for (size_t i = 0; i < idx.size(); ++i)
    {
        cv::morphologyEx(imgs[i], imgs[i], cv::MORPH_OPEN, kernel);
        cv::dilate(imgs[i], imgs[i], kernel);
        cv::GaussianBlur(imgs[i], imgs[i], cv::Size(5, 5), 0);
    

        // Resize to 28 x 28 to match the MNIST dataset
        resize(imgs[i], imgs[i], targetSize);
    }

    // Display Images
    for (size_t i = 0; i < idx.size(); ++i)
    {
        cv::namedWindow("Display Image3", cv::WINDOW_KEEPRATIO );
        cv::imshow("Display Image3", imgs[i]);
        cv::waitKey(0);
    }
    cv::destroyAllWindows();
    return 0;
}