#ifndef UTILITY_HPP
#define UTILITY_HPP
#include <opencv2/opencv.hpp>
#include <stdio.h>

using namespace cv;

#define DEBUG 1

/*
 * @brief: ifdef DEBUG, show the image
 */
void ShowImage(const char *win_name, const Mat &img);


/*
 * @brief: if define DEBUG, save the image.
 */
void SaveImage(const char *name, Mat &img);

/*
 * @brief: log
 */
void Log_d(const char *format, ...);

#define Log(fmt, args...) do { \
    printf("%s ", __func__); \
    Log_d(fmt, ##args); \
} while(0)

#endif //UTILITY_HPP
