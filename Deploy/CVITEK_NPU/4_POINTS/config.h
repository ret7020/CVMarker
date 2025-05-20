#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d.hpp>

#define MODEL_SCALE 0.0039216
#define MODEL_MEAN 0.0
#define MODEL_CLASS_CNT 4
#define MODEL_THRESH 0.15
#define MODEL_NMS_THRESH 0.5
#define FRAME_SIZE 640
#define STREAM_VISUALIZATION
// #define DEBUG_SAVE_ON_FAIL

#define CENTER_COLOR cv::Scalar(200, 100, 30)

double marker_size = 54.0; // mm

cv::Scalar color_map[4] = {cv::Scalar(255, 0, 0),
                           cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
                           cv::Scalar(100, 100, 100)};


// Calibrated via checkerboard (whoop camera)
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 736.93767102, 0, 304.55131639,
                         0, 733.53876077, 218.97819259,
                         0, 0, 1);
cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << -0.77879963, 1.16676111, 0.00648388, 0.00149256, -1.26450536);

std::vector<cv::Point3f> object_points = {
    cv::Point3f(-marker_size / 2, -marker_size / 2, 0),
    cv::Point3f(marker_size / 2, -marker_size / 2, 0),
    cv::Point3f(marker_size / 2, marker_size / 2, 0),
    cv::Point3f(-marker_size / 2, marker_size / 2, 0)};
