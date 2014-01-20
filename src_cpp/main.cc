#include "detect_plates.hpp"
#include "parse_chars.hpp"
#include "judge_plate.hpp"
#include "judge_chars.hpp"
#include <opencv2/opencv.hpp>
#include "utility.hpp"
#include <stdio.h>
#include <iostream>
using namespace std;
using namespace cv;

void detect_plates(Mat img_plate, Mat img_contain_plates) {

    vector<Mat> img_plate_regions;
    vector<Rect> plate_regions;
    FindPlates(img_plate, img_plate_regions, plate_regions);
   // for (int i = 0; i < plate_regions.size(); i++) {
   //     rectangle(img_contain_plates, plate_regions[i], Scalar(255, 0, 0), 2);
   // }

    CharRecognizer *char_recognizer = CharRecognizer::GetCharRecognizer();
    bool is_init = char_recognizer->Init();

    for (int k = 0; k < img_plate_regions.size(); k++) {
    	Mat p = img_plate_regions[k].reshape(1,1);
	p.convertTo(p, CV_32FC1);
	if (!JudgePlate(p)) {
		continue;
	}
	rectangle(img_contain_plates, plate_regions[k], Scalar(255, 0, 0), 2);
		
    	vector<Mat> img_chars = SegmentChars(img_plate_regions[k]);
    	for (int m = 0; m < img_chars.size(); m++) {

    		Mat feature = GetFeatures(img_chars[m], 15);
    		char res = char_recognizer->Classify(feature);
    		char sz_res[10];
    		sprintf(sz_res, "%c", res);
    		putText(img_contain_plates, sz_res, Point(30 + m * 30, 50), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0));
     	}
    }
}


int main(int argc, char *argv[]) {
    printf("load image");
    std::cout << "load image test" << std::endl;
    Mat img_plate = imread(argv[1]);
    Mat img_contain_plate = img_plate.clone();
    detect_plates(img_plate, img_contain_plate);
    imwrite("1.png", img_contain_plate);
    ShowImage("b", img_contain_plate);
    return 0;
}
