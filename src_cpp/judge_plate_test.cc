#include "judge_plate.hpp"
#include "utility.hpp"

using namespace cv;

int main(int argc, char *argv[]) {
  Mat m = imread(argv[1]);
  Mat img1;
  if (m.channels() == 3) {
    cvtColor(m, img1, CV_RGB2GRAY);
  } else {
    img1 = m;
  }
  Mat img =  img1.reshape(1,1);
  img.convertTo(img, CV_32FC1);
  Log("cc %d, col %d, row %d\n",img.channels(), img.cols, img.rows);
  bool b = JudgePlate(img);
  Log("ret is %d\n", b);
  return 0;
}
