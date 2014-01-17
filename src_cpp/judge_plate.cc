#include <opencv2/opencv.hpp>

#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

PlateJudger* PlateJudger::judger = new PlateJudger();

PlateJudger* PlateJudger::GetPlateJudger() {
  if (judger == NULL) {
    judger = new PlateJudger();
  }
  return judger;
}

PlateJudger::PlateJudger() {
  is_trained = false;
  svm_params.svm_type = CvSVM::C_SVC;
  svm_params.kernel_type = CvSVM::LINEAR;
  svm_params.degree = 0;
  svm_params.gamma = 1;
  svm_params.coef0 = 0;
  svm_params.C = 1;
  svm_params.nu = 0;
  svm_params.p = 0;
  svm_params.term_crit = cvTermCriteria(CV_TERMCRIT_ITER, 1000, 0.01);
}

bool PlateJudger::Train(Mat train_data, Mat mat_classes) {
  if (is_trained == false) {
    svm_classifier.train(train_data, mat_classes, Mat(), Mat(), svm_params);
  }
  is_trained = true;
  return is_trained;
}

int PlateJudger::Predict(Mat img) {
  int resp = -1;
  if (is_trained) {
    resp = (int)svm_classifier.predict(img);
  }
  return resp;
}

bool PlateJudger::LoadData(const char *svm_xml) {
  FileStorage fs;
  fs.open(svm_xml, FileStorage::READ);
  if (!fs.isOpened()) {
    return false;
  }

  Mat train_data;
  Mat mat_classes;
  fs["TrainingData"] >> train_data;
  fs["classes"] >> mat_classes;

  if (is_trained == false) {
    is_trained = Train(train_data, mat_classes);
  }
  return is_trained;
}
  
bool JudgePlate(Mat img) {
  bool is_plate = false;
  PlateJudger *judger = PlateJudger::GetPlateJudger();
  judger->LoadData("SVM.xml");
  int resp = judger->Predict(img);
  if (resp == 1)
    is_plate = true;
  
  return is_plate;
}
