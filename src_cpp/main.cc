#include "detect_plates.hpp"
#include <opencv2/opencv.hpp>
#include "utility.hpp"
#include <stdio.h>
#include <iostream>
using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {
    printf("load image");
    std::cout << "load image test" << std::endl;
    Mat img_plate = imread(argv[1]);
    printf("1");
    //ShowImage("a", img_plate);
    printf("hello world");
    vector<Mat> img_plate_regions;
    vector<Rect> plate_regions;
    printf("hello world");
    FindPlates(img_plate, img_plate_regions, plate_regions);
    for (int i = 0; i < plate_regions.size(); i++) {
        rectangle(img_plate, plate_regions[i], Scalar(255, 0, 0), 2);

    }
    ShowImage("b", img_plate);
    return 0;
}
