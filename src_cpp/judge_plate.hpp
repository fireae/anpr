#ifndef JUDGE_PLATE_HPP
#define JUDGE_PLATE_HPP
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

class PlateJudger {
 public:
  static PlateJudger *GetPlateJudger();
  bool Train(Mat train_data, Mat mat_classes);
  int Predict(Mat img);
  bool LoadData(const char *svm_xml);
 private:
  bool is_trained;
  CvSVMParams svm_params;
  CvSVM svm_classifier;
  
 private:
  PlateJudger();
  ~PlateJudger();
  static PlateJudger *judger;
};

bool JudgePlate(Mat img);

#endif
