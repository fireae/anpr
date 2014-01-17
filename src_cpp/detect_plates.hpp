#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
using namespace std;
using namespace cv;

/*
 *  @brief: verify the candidate plate size
 */
bool VerifySize(RotatedRect rr);

/*
 *  @brief: preprocess to the image, blur->sobel-> threshold->mophologyex close
 */
Mat PreprocessImage(Mat img_plate);

/*
 * @brief: use floorFill for the candidate plate region
 */
bool FloorPlate(Mat img_plate, RotatedRect &candidate_rect, Mat &candidate_plate);

/*
 * @brief: find the possible plate region from the image
 */
vector<RotatedRect> FindCandidatePlate(Mat img_plate);

/*
 * @brief: find plate images and plate regions from the image_plate
 */
void FindPlates(Mat img_plate, vector<Mat> &img_plate_regions, vector<Rect> &plate_regions);

/*
 * @brief: crop the select region from the img_plate
 */
Mat CropImage(Mat img_plate, RotatedRect rr);

/*
 * @brief: histogram equation for the image
 */
Mat HistEqu(Mat img);
