#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

using namespace std;
using namespace cv;

/*--------------------------------------functions for task-1 sobel-----------------------------------*/

Mat convolve(Mat img, vector<vector<float>> filter, int stride = 1) {

    int kernelSize = filter.size();
    Mat gradOut(img.rows, img.cols, CV_8U, Scalar(0));// empty image to calculate gradient on it

    int rowsOut = ((img.rows - kernelSize) / stride) + 1; // calculate number of output rows 
    int colsOut = ((img.cols - kernelSize) / stride) + 1; // calculate number of output columns

    // loop on the image 
    for (int r = 0; r < rowsOut; r++)
        for (int c = 0; c < colsOut; c++)
        {
            float grad = 0;
            // loop on the filter 
            for (int i = 0; i < kernelSize; i++)
                for (int j = 0; j < kernelSize; j++)
                {
                    grad += img.at<uchar>(i + (r * stride), j + (c * stride)) * filter[i][j];
                }

            // set the pixel value to the result of gradient  
            gradOut.at<uchar>((r * stride) + int(kernelSize / 2), (c * stride) + int(kernelSize / 2))
                = saturate_cast<uchar>(abs(grad));

        }

    return gradOut;
}

void boxBlur(Mat src, Mat& dst, int stride = 1) {

    vector<vector<float>> filter = { {0.11,0.11,0.11},
                                    {0.11,0.11,0.11},
                                    {0.11,0.11,0.11} }; // 0.11 = 1/9 to avg pixels on 3*3 window


    dst = convolve(src, filter, stride);


}

void apply_sobel(Mat src, Mat& dst, int stride = 1) {

    // 3*3 kernel on X-axis
    vector<vector<float>> kernelX = { {-1,0,1},
                                   {-2,0,2},
                                   {-1,0,1} };

    // 3*3 kernel on Y-axis
    vector<vector<float>> kernelY = { {1,2,1},
                                    {0,0,0},
                                    {-1,-2,-1} };


    Mat sobelX = convolve(src, kernelX, stride);//convolve on image with kernelX to get vertical edges 
    Mat sobelY = convolve(src, kernelY, stride);//convolve on image with kernelY to get horizontal edges

    Mat gradOut(src.rows, src.cols, CV_8U, Scalar(0));// empty image to calculate gradient of X,Y on it

    //loop on gradient X,Y images to get vertical and horizontal edges on one image sqrt(gradX^2 + gradY^2)
    for (int i = 0; i < src.rows; i++)
        for (int j = 0; j < src.cols; j++) {
            uchar gradX = sobelX.at<uchar>(i, j);
            uchar gradY = sobelY.at<uchar>(i, j);
            gradOut.at<uchar>(i, j) = saturate_cast<uchar>(sqrt((gradX * gradX) + (gradY * gradY)));

        }

    dst = gradOut;
}

void sobel_3channels(Mat src, Mat& dst, int stride = 1) {
   
    Mat rgb[3], merged;

    cvtColor(src, src, COLOR_BGR2RGB); // convert image from BGR to RGB

    split(src, rgb); // split image to seperated 3 channels 

    vector<Mat> channels_vector;

    // loop on every channel and apply boxBlur then apply sobel 
    for (int i = 0; i < 3; i++)
    {
        Mat img_blur, channel;
        boxBlur(rgb[i], img_blur);
        apply_sobel(img_blur, channel, stride);
        channels_vector.push_back(channel);
    }

    merge(channels_vector, merged); // merge 3 channels to one colored image 

    dst = merged;
}

void task1() {

   // read image
   Mat img_colored = imread("image-1.png", IMREAD_COLOR);

   // Convert to grayscale
   Mat img_gray;
   cvtColor(img_colored, img_gray, COLOR_BGR2GRAY);

   // apply an avg filter (box blur) on the image
   Mat img_blur;
   boxBlur(img_gray, img_blur);//takes 3 parameters-->source image,destination image,stride

   // apply sobel on grayscale image (1 channel)
   Mat sobel_gray;
   apply_sobel(img_gray, sobel_gray, 3);//takes 3 parameters-->source image,destination image,stride 

   // apply sobel on blured grayscale image
   Mat sobel_blured;
   apply_sobel(img_blur, sobel_blured, 3);

   // apply sobel on RGB image (3 channels)
   Mat sobel_rgb;
   sobel_3channels(img_colored, sobel_rgb, 3);//takes 3 parameters-->source image,destination image,stride

   // Save images
   imwrite("Sobel on gray Image.png", sobel_gray);
   imwrite("Sobel on blured gray Image.png", sobel_blured);
   imwrite("Sobel on colored Image.png", sobel_rgb);

   // Display Sobel edge detection images
   imshow("Sobel on gray Image", sobel_gray);
   imshow("Sobel on blured gray Image", sobel_blured);
   imshow("Sobel on colored Image", sobel_rgb);

}

/*---------------------------------functions for task-2 filter orange--------------------------------*/

void filterColor(Mat src, Mat& dst, int startRange[3], int endRange[3]) {

    Mat rgb[3], merged;

    cvtColor(src, src, COLOR_BGR2RGB); // convert image from BGR to RGB

    split(src, rgb); // split image to seperated 3 channels 

    vector<Mat> channels_vector = { rgb[0],rgb[1],rgb[2] };

    // loop on every pixel and check the 3 values for this pixel if in range then remove it
    for (int r = 0; r < src.rows; r++)
        for (int c = 0; c < src.cols; c++)
        {
            int inRange = 0;
            for (int i = 0; i < 3; i++)
            {
                int val = int(channels_vector[i].at<uchar>(r, c));
                if (val >= startRange[i] && val <= endRange[i])
                    inRange += 1;

            }

            if (inRange == 3)
            {
                channels_vector[0].at<uchar>(r, c) = 0;
                channels_vector[1].at<uchar>(r, c) = 0;
                channels_vector[2].at<uchar>(r, c) = 0;
            }

        }



    merge(channels_vector, merged); // merge 3 channels to one colored image 

    cvtColor(merged, merged, COLOR_RGB2BGR); // convert image from RGB to BGR

    dst = merged;

}

void task2()
{
    // read image
    Mat img = imread("image-1.png", IMREAD_COLOR);

    
    int start[3] = { 220,127,39 }; // Minimum range for orange color as RGB
    int end[3] = { 255,170,111 }; //  Maximum range for orange color as RGB

    // apply filter method to remove orange color
    Mat filtered;
    filterColor(img, filtered, start, end);

    // save image
    imwrite("filter orange(task2).png", filtered);

    // display image
    imshow("filter orange(task2)", filtered);



}

/*--------------------------------------------Main---------------------------------------------------*/

int main(int argc, char* argv[])
{

    task1(); // run task 1 (sobel edge detection)
    task2(); // run task 2 (filter orange color)

    waitKey(0);
    destroyAllWindows();

    return 0;
}
