#include "judge_chars.hpp"
#include "parse_chars.hpp"
#include "utility.hpp"

int main(int argc, char *argv[]) {
  Mat img_char = imread(argv[1], 0);
  ShowImage("char", img_char);
  Mat img_gray;
  if (img_char.channels() == 3) {
    cvtColor(img_char, img_gray, CV_RGB2GRAY);
  } else {
    img_gray = img_char;
  }
  ShowImage("gray", img_gray);
  
  Mat img_char_bin;
  threshold(img_gray, img_char_bin, 60, 255, THRESH_BINARY_INV);
  Mat img_ok_char = RectifyImage(img_char_bin);
  ShowImage("img", img_ok_char);
  Mat feature = GetFeatures(img_ok_char, 15);

  CharRecognizer *recognizer = CharRecognizer::GetCharRecognizer();
  if (recognizer->Init() == false) {
    Log("init failed");
  }

  int res = recognizer->Classify(feature);
  Log("result is : %d\n", res);
  return 0;
}
