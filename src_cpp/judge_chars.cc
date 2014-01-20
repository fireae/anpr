#include "judge_chars.hpp"
#include "utility.hpp"


Mat ProjectHist(Mat img, int t) {

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
  Log("sz is %d\n", sz);
  return hist;
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

Mat GetFeatures(Mat in, int sizeData) {
    Log("in image is \n");
    ShowImage("in is", in);
    //Histogram features
    Mat vhist=ProjectHist(in, 0);
    Mat hhist=ProjectHist(in, 1);
    Log("vc %d, vr %d\n", vhist.cols, vhist.rows);
    Log("hc %d, hr %d\n", hhist.cols, hhist.rows);
    //Low data feature
    Mat lowData;
    resize(in, lowData, Size(sizeData, sizeData) );

    //Last 10 is the number of moments components
    int numCols=vhist.cols+hhist.cols+lowData.cols*lowData.cols;
    Log("numCols is %d\n", numCols);
    Mat out=Mat::zeros(1,numCols,CV_32F);
    //Asign values to feature
    int j=0;
    for(int i=0; i<vhist.cols; i++)
    {
        out.at<float>(j)=vhist.at<float>(i);
        j++;
    }
    for(int i=0; i<hhist.cols; i++)
    {
        out.at<float>(j)=hhist.at<float>(i);
        j++;
    }
    for(int x=0; x<lowData.cols; x++)
    {
        for(int y=0; y<lowData.rows; y++){
            out.at<float>(j)=(float)lowData.at<unsigned char>(x,y);
            j++;
        }
    }
   
    return out;
}

//////////////////////////
// CharRecognizer

int CharRecognizer::num_chars = 30;
const char CharRecognizer::chars[] = {
    '0','1','2','3','4','5','6','7','8','9',
    'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K',
    'L', 'M', 'N', 'P', 'R', 'S', 'T', 'V',
    'W', 'X', 'Y', 'Z'};

CharRecognizer::CharRecognizer() {
  is_trained = false;
}

CharRecognizer::~CharRecognizer() {
}

CharRecognizer * CharRecognizer::recognizer = new CharRecognizer();

CharRecognizer * CharRecognizer::GetCharRecognizer() {
  if (recognizer == NULL)
    recognizer = new CharRecognizer();
  return recognizer;
}

bool CharRecognizer::Init() {
  Log("first init");
  if (is_trained == false) {
    Mat train_data;
    Mat data_classes;
    if (LoadTrainData(train_data, data_classes)) {
      Train(train_data, data_classes, 10);
      is_trained = true;
    }
  }
  return is_trained;
}

bool CharRecognizer::LoadTrainData(Mat &train_data, Mat &data_classes) {
  FileStorage fs;
  fs.open("OCR.xml", FileStorage::READ);
  if (!fs.isOpened()) {
    Log("ocr xml load failed");
    return false;
  }

  fs["TrainingDataF15"] >> train_data;
  fs["classes"] >> data_classes;
  Log("load data\n");
  if (train_data.empty() || data_classes.empty()) {
    return false;
  }

  return true;
}

void CharRecognizer::Train(Mat train_data, Mat data_classes, int num_layer) {
  Log("train begin");
  Mat layer(1, 3, CV_32SC1);
  layer.at<int>(0) = train_data.cols;
  layer.at<int>(1) = num_layer;
  layer.at<int>(2) = num_chars;
  Log("0: %d, 1: %d, 2:%d \n", layer.at<int>(0), layer.at<int>(1), layer.at<int>(2));
  ann.create(layer, CvANN_MLP::SIGMOID_SYM, 1.0, 1.0);

  // prepare trainclasses
  Mat train_classes;
  train_classes.create(train_data.rows, num_chars, CV_32FC1);
  for (int i = 0; i < train_classes.rows; i++) {
    for (int k = 0; k < train_classes.cols; k++) {
      if (k == data_classes.at<int>(i)) {
        train_classes.at<float>(i, k) = 1;
      } else {
        train_classes.at<float>(i, k) = 0;
      }
    }
  }
  Log("train data type %d\n", train_data.type());
  train_data.convertTo(train_data, CV_32FC1);
  Mat weights(1, train_classes.rows, CV_32FC1, Scalar::all(1));
  ann.train(train_data, train_classes, weights);
  is_trained = true;
  Log("train failed %d\n", is_trained);
}

char CharRecognizer::Classify(Mat img) {
  Log("Classify begin \n");

  int result = -1;
  if (is_trained) {
    Log("classify is now\n");
    Mat output(1, num_chars, CV_32FC1);
    Log("img type: %d\n", img.type());
    Log("img c %d, r %d\n", img.cols, img.rows);
    ann.predict(img, output);

    Point max_loc;
    double max_value;
    minMaxLoc(output, 0, &max_value, 0, &max_loc);
    Log("max value %f, x %d, y %d", max_value, max_loc.x, max_loc.y);
    result = max_loc.x;
  }

  return chars[result];
}
