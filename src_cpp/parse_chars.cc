#include "parse_chars.hpp"
#include "utility.hpp"

const int kCharSize = 20;

Mat PreprocessImage(Mat img) {
  ShowImage("a", img);
  Mat gray;
  if (img.channels() == 3) {
    cvtColor(img, gray, CV_RGB2GRAY);
  } else {
    gray = img;
  }

  Mat img_bin = gray.clone();
  threshold(gray, img_bin, 60, 255, CV_THRESH_BINARY_INV);
  ShowImage("b", img_bin);

  return img_bin;
}

bool VerifySize(Mat img) {
  // char size 16 * 30
  bool is_ok = false;
  float aspect = 16.0f / 30.0f;
  float char_aspect = (float)img.cols/(float)img.rows;
  float error = 0.4;
  float min_height = 8;
  float max_height = 40;
  float min_aspect = 0.2;
  float max_aspect = aspect + aspect*error;

  float pix_cnt = countNonZero(img);
  float area = img.cols * img.rows;
  float per_pixel = pix_cnt / area;

  if ( (per_pixel < 0.8) &&
       (char_aspect > min_aspect) &&
       (char_aspect < max_aspect) &&
       (img.rows >= min_height) &&
       (img.rows < max_height)) {
    is_ok = true;
  }
  
  return is_ok;
}

vector<Rect> FindChars(Mat img) {
  ShowImage("findchar", img);
  
  vector<Rect> candidate_rects;
  vector< vector<Point> > contours;

  findContours(img, contours, CV_RETR_LIST, CV_CHAIN_APPROX_NONE);

  vector< vector<Point> >::iterator it = contours.begin();
  vector< vector<Point> >::iterator ite = contours.end();
  Log("pos contours %d\n", contours.size());

  while (it < ite) {
    RotatedRect rr = minAreaRect(*it);
    Rect r = rr.boundingRect();
    Log("r.x %d r.y %d r.w %d r.h %d\n", r.x, r.y, r.width, r.height);
    if (r.x + r.width > img.cols || r.y + r.height > img.rows ||
        r.x < 0 || r.y < 0) {
      it++;
      continue;
    }
    Mat mr = img(r);
    ShowImage("char", mr);
    if (VerifySize(mr)) {
      candidate_rects.push_back(r);
      ShowImage("canchar", mr);
    }
    it++;
  }
  Log("candidate_rects size %d\n", candidate_rects.size());
  return candidate_rects;
}

Mat RectifyImage(Mat img) {
  ShowImage("recimg", img);
  int max_len = (img.rows > img.cols) ? img.rows:img.cols;
  Mat rot_mat = Mat::eye(2,3,CV_32FC1);
  rot_mat.at<float>(0, 2) = max_len/2 - img.cols/2;
  rot_mat.at<float>(1, 2) = max_len/2 - img.rows/2;

  Mat img_warp(max_len, max_len, img.type());
  warpAffine(img, img_warp, rot_mat, img_warp.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0));

  ShowImage("warp", img_warp);
  Mat output;
  resize(img_warp, output, Size(kCharSize, kCharSize));
  ShowImage("output", output);
  
  return output;
}

vector<Mat> SegmentChars(Mat img) {
  vector<Mat> img_chars;
  Mat img_process = PreprocessImage(img);
  ShowImage("proc", img_process);

  vector<Rect> rect_chars = FindChars(img_process);
  Log("chars size: %d\n", rect_chars.size());
  int i = 0;
  for (i = 0; i < rect_chars.size(); i++) {
    Mat m = img(rect_chars[i]);
    Mat img_rectify = RectifyImage(m);
    img_chars.push_back(img_rectify);
  }

  return img_chars;
}

Mat ProjectHist(Mat img, int t) {
  Mat feature;
  int sz = (t)? img.rows : img.cols;
  Mat hist=Mat::zeros(1, sz, CV_32F);
  for (int i = 0; i < sz; i++) {
    Mat data = (t)? img.row(i): img.col(i);
    hist.at<float>(i) = countNonZero(data);
  }

  double min_v = 0.0;
  double max_v = 0.0;
  minMaxLoc(hist, &min_v, &max_v);
  
  if (max_v > 1.0) {
    hist.convertTo(hist, -1, 1.0f/max_v, 0);
  }

  return feature;
}

Mat MakeFeatures(Mat img, int sz_data) {

  Mat features;
  Mat hrow = ProjectHist(img, 0);
  Mat hcol = ProjectHist(img, 1);

  // low data feature
  Mat low_data;
  resize(img, low_data, Size(sz_data, sz_data));

  int num_cols = hrow.cols + hcol.cols + low_data.cols * low_data.cols;

  features = Mat::zeros(1, num_cols, CV_32F);
  int j = 0;
  for (int i = 0; i < hrow.cols; i++) {
    features.at<float>(j) = hrow.at<float>(i);
    j++;
  }
  for (int i = 0; i < hcol.cols; i++) {
    features.at<float>(j) = hcol.at<float>(i);
    j++;
  }
  for (int k = 0; k < low_data.cols; k++) {
    for (int m = 0; m < low_data.rows; m++) {
      features.at<float>(j) = low_data.at<unsigned char>(k, m);
    }
  }
    
  return features;
}
