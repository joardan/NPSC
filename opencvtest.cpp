#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

using namespace cv;
 
int main(int argc, char** argv )
{
  Eigen::MatrixXd m(2,2);
  m(0,0) = 3;
  m(1,0) = 2.5;
  m(0,1) = -1;
  m(1,1) = m(1,0) + m(0,1);
  std::cout << m << std::endl;
 if ( argc != 2 )
 {
 printf("usage: opencvtest.out <Image_Path>\n");
 return -1;
 }
 
 Mat image;
 image = imread( argv[1], IMREAD_COLOR );
 
 if ( !image.data )
 {
 printf("No image data \n");
 return -1;
 }
 namedWindow("Display Image", WINDOW_AUTOSIZE );
 imshow("Display Image", image);
 
 waitKey(0);
 
 return 0;
}