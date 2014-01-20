#include "utility.hpp"
#include <stdarg.h>
#include <stdio.h>

void Log_d(const char *format, ...) {
#if DEBUG
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
#endif
}


void SaveImage(const char *name, Mat &img) {
#if DEBUG
    imwrite(name, img);
#endif
}

void ShowImage(const char *win_name, const Mat &img) {
#if DEBUG
    namedWindow(win_name);
    imshow(win_name, img);
    waitKey(0);
#endif
}
