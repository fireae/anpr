#ifndef PARSE_CHARS_HPP
#define PARSE_CHARS_HPP

#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>

using namespace cv;
using namespace std;

// preprocess
Mat PreprocessImage(Mat img);

// verify the size of the candidate char region
bool VerifySize(Mat img);

// find the possible image region from the image.
vector<Rect> FindChars(Mat img);

// find the possible char image from the image
vector<Mat> SegmentChars(Mat img);

// adjust the image
Mat RectifyImage(Mat img);


#endif // PARSE_CHARS_HPP
