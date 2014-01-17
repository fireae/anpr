#include "detect_plates.hpp"
#include "utility.hpp"
#include <stdarg.h>


bool VerifySize(RotatedRect rr) {
    Log("rr is w %f, h %f\n", rr.size.width, rr.size.height);
    float error = 0.4;
    const float aspect = 4.8;
    int max_area = 120 * 120 * aspect;
    int min_area = 15 * 15 * aspect;

    float min_rate = aspect - aspect * error;
    float max_rate = aspect + aspect * error;
    int area = rr.size.width * rr.size.height;
    float rate = (float)rr.size.width / (float)rr.size.height;
    if (rate < 1)
        rate = 1.0 / rate;
    if (area > max_area || area < min_area || rate < min_rate || rate > max_rate)
        return false;
    return true;
}

Mat PreprocessImage(Mat img_plate) {
    // convert to gray
    Mat img_gray;
    if (img_plate.channels() == 3) {
        cvtColor(img_plate, img_gray, CV_BGR2GRAY);
    } else {
        img_plate.copyTo(img_gray);
    }
    ShowImage("gray", img_gray);

    // blur and sobel
    blur(img_gray, img_gray, Size(5,5));
    Mat img_sobel;
    Sobel(img_gray, img_sobel, CV_8U, 1, 0, 3, 1, 0);

    // convert to Bin
    Mat img_bin;
    threshold(img_sobel, img_bin, 0, 255, CV_THRESH_OTSU+CV_THRESH_BINARY);

    // erode
    //Mat img_erode;
    //erode(img_bin, img_erode, cv::Mat());
    // close morphology
    Mat element = getStructuringElement(MORPH_RECT, Size(17, 3));
    morphologyEx(img_bin, img_bin, CV_MOP_CLOSE, element);

    return img_bin;
}

Mat HistEqu(Mat img) {
    Mat result(img.size(), img.type());
    if (img.channels() == 3) {
        vector<Mat> hsv_split;
        Mat hsv;
        cvtColor(img, hsv, CV_BGR2HSV);
        split(hsv, hsv_split);
        equalizeHist(hsv_split[2], hsv_split[2]);
        merge(hsv_split, hsv);
        cvtColor(hsv, result, CV_HSV2BGR);
    } else {
        equalizeHist(img, result);
    }
    return result;
}

Mat CropImage(Mat img_plate, RotatedRect rr) {
    Log("crop rr w :%f, h :%f\n", rr.size.width, rr.size.height);
    float rate = (float)rr.size.width/(float)rr.size.height;
    float angle = rr.angle;
    if (rate < 1)
        angle = angle + 90;

    Mat rotate_mat = getRotationMatrix2D(rr.center, angle, 1);
    Mat img_rotated;
    warpAffine(img_plate, img_rotated, rotate_mat, img_plate.size());
    ShowImage("rotated", img_rotated);

    // crop image
    Size rect_size = rr.size;
    if (rate < 1) {
        int t = rect_size.width;
        rect_size.width = rect_size.height;
        rect_size.height = t;
    }
    Log("crop rr2 w :%f, h :%f\n", rr.size.width, rr.size.height);

    Mat img_crop;
    getRectSubPix(img_rotated, rect_size, rr.center, img_crop);
    ShowImage("crop", img_crop);

    Mat img_crop_color;
    if (img_crop.channels() == 1) {
        cvtColor(img_crop, img_crop_color, CV_GRAY2BGR);
    } else {
        img_crop.copyTo(img_crop_color);
    }

    Mat result_resized;
    result_resized.create(33, 144, CV_8UC3);
    resize(img_crop_color, result_resized, result_resized.size(), 0, 0, INTER_CUBIC);

    Mat gray;
    cvtColor(result_resized, gray, CV_BGR2GRAY);
    gray = HistEqu(gray);
    blur(gray, gray, Size(3,3));

    return gray;
    ShowImage("img_hist_gray", gray);
}

bool FloorPlate(Mat img_plate, RotatedRect &candidate_rect, Mat &candidate_plate) {
    Mat result;

    int low_diff = 30;
    int up_diff = 30;
    int connectivity = 4;
    int new_mask_val = 255;
    int num_seeds = 10;
    srand(time(NULL));

	int min_size = (candidate_rect.size.width < candidate_rect.size.height)?
		    candidate_rect.size.width : candidate_rect.size.height;
    min_size = min_size * 0.5;
    Mat mask;
	mask.create(img_plate.rows+2, img_plate.cols+2, CV_8UC1);
    mask = Scalar::all(0);
	Rect ccomp;
	int flags = connectivity + (new_mask_val << 8) +
		CV_FLOODFILL_FIXED_RANGE + CV_FLOODFILL_MASK_ONLY;

	for (int k = 0; k < num_seeds; k++) {
	    Point seed;
	    seed.x = candidate_rect.center.x + rand()%(int)min_size - (min_size/2);
	    seed.y = candidate_rect.center.y + rand()%(int)min_size - (min_size/2);
	    int area = floodFill(img_plate, mask, seed, Scalar(255, 0, 0),
		                 &ccomp, Scalar(low_diff, low_diff, low_diff),
		                 Scalar(up_diff, up_diff, up_diff), flags);
	}

    ShowImage("mask", mask);
	vector<Point> points_interest;
	Mat_<uchar>::iterator it_mask = mask.begin<uchar>();
	Mat_<uchar>::iterator it_end = mask.end<uchar>();

	for (; it_mask != it_end; it_mask++) {
	    if (*it_mask == 255)
            points_interest.push_back(it_mask.pos());
	}

	RotatedRect rr = minAreaRect(points_interest);
	if (VerifySize(rr)) {
	    candidate_plate = CropImage(img_plate, rr);
        candidate_rect = rr;
        ShowImage("cp", candidate_plate);
		return true;
	}

	return false;
}

vector<RotatedRect> FindCandidatePlate(Mat img_plate_bin) {
    vector<RotatedRect> out_rects;
    std::cout << out_rects.size() << endl;
    vector< vector<Point> > contours;
    findContours(img_plate_bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

    Mat img_show = img_plate_bin.clone();
    drawContours(img_show, contours, -1, Scalar(255));
    ShowImage("contours", img_show);
    Log("contours size is %d\n", (int)contours.size());

    vector<vector<Point> >::iterator it = contours.begin();
    while (it != contours.end()) {
        RotatedRect rr = minAreaRect(Mat(*it));
        if (VerifySize(rr)) {
            out_rects.push_back(rr);
        }
        it++;
    }

    Log("out rect size: %d\n", out_rects.size());
    return out_rects;
}
void FindPlates(Mat img_plate, vector<Mat> &img_plate_regions, vector<Rect> &plate_regions) {

    Mat img_plate_bin = PreprocessImage(img_plate);
    ShowImage("bin", img_plate_bin);

    vector<RotatedRect> candidate_regions = FindCandidatePlate(img_plate_bin);
    Log("candidate regions : %d\n", candidate_regions.size());
    for (int i = 0; i < candidate_regions.size(); i++) {
        Rect r = candidate_regions[i].boundingRect();
        rectangle(img_plate, r, Scalar(255), 2);
    }
    ShowImage("candi", img_plate);

    bool is_plate = false;
	for (int i =0 ; i < candidate_regions.size(); i++) {
        Mat candidate_plate;
        is_plate = FloorPlate(img_plate, candidate_regions[i], candidate_plate);
		if (is_plate) {
            Log("candi_plate w %d, h %d\n", candidate_plate.cols, candidate_plate.rows);
			img_plate_regions.push_back(candidate_plate);	
            ShowImage("candidate_plate", candidate_plate);

            Rect rect = candidate_regions[i].boundingRect();
            plate_regions.push_back(rect);
            Log("candidate rect w : %d, h: %d\n", rect.width, rect.height);
		}
	}

}
