#ifndef JUDGE_CHARS_HPP
#define JUDGE_CHARS_HPP

#include <opencv2/opencv.hpp>
using namespace cv;

Mat ProjectHist(Mat img, int t);

Mat MakeFeatures(Mat img, int sz_data);

Mat GetFeatures(Mat in, int sizeData);

class CharRecognizer {
 public:
  static CharRecognizer *GetCharRecognizer();

  bool Init();
  
  void Train(Mat train_data, Mat data_classes, int layer_num);

  int Classify(Mat img);

  bool LoadTrainData(Mat &train_data, Mat &data_classes);

 private:
  static CharRecognizer *recognizer;

  CvANN_MLP ann;
  bool is_trained;
  static int num_chars;
  static const char chars[50];
  
 private:
  CharRecognizer();
  ~CharRecognizer();
  
};

#endif
