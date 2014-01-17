#include "parse_chars.hpp"

int main(int argc, char *argv[]) {
  Mat m = imread(argv[1]);

  vector<Mat> img_chars = SegmentChars(m);
  for (int i = 0; i < img_chars.size(); i++) {
    namedWindow("a");
    imshow("a", img_chars[i]);
    waitKey();
  }
  
  return 0;
}
