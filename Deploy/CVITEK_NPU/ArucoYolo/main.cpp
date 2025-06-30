#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "config.h"
#include "MJPEGWriter.h"


float markerSize_ = 100; // pixels
std::vector<cv::Point2f> dstPts = {
    cv::Point2f(0, 0),
    cv::Point2f(markerSize_ - 1, 0),
    cv::Point2f(markerSize_ - 1, markerSize_ - 1),
    cv::Point2f(0, markerSize_ - 1)
};
cv::aruco::Dictionary dictionary;
cvitdl_handle_t tdl_handle = nullptr;

MJPEGWriter test;

volatile uint8_t interrupted = 0;
void interrupt_handler(int signum) {
    printf("Signal: %d\n", signum);
    interrupted = 1;
}

cv::Mat extractMarkerBits(const cv::Mat& markerImage) {
    cv::Mat bits(5, 5, CV_8UC1); // 5x5 matrix, 8-bit unsigned, 1 channel
    int size = markerImage.rows;
    int cellSize = size / 7;

    for (int y = 1; y < 6; y++) { // Skip the black border
        for (int x = 1; x < 6; x++) {
            int startX = x * cellSize;
            int startY = y * cellSize;

            cv::Mat cell = markerImage(cv::Rect(startX, startY, cellSize, cellSize));
            cv::Scalar meanVal = cv::mean(cell);

            if (meanVal[0] > 127) {
                bits.at<uchar>(y - 1, x - 1) = 1;
            } else {
                bits.at<uchar>(y - 1, x - 1) = 0;
            }
        }
    }
    return bits;
}

void initModel(char * modelFilePath) {
    CVI_S32 ret = CVI_TDL_CreateHandle(&tdl_handle);
    if (ret != CVI_SUCCESS) {
        throw std::runtime_error("Create TDL handle failed with error code: " + std::to_string(ret));
    }
    // setup preprocess
    InputPreParam preprocess_cfg = CVI_TDL_GetPreParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION);
    for (int i = 0; i < 3; i++) {
      preprocess_cfg.factor[i] = 0.003922;
      preprocess_cfg.mean[i] = 0.0;
    }
    preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;
    ret = CVI_TDL_SetPreParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, preprocess_cfg);
    if (ret != CVI_SUCCESS) {
        std::ostringstream errorMsg;
        throw std::runtime_error("Can not set yolov8 preprocess parameters" + std::to_string(ret));
    }
    // setup yolo algorithm preprocess
    cvtdl_det_algo_param_t yolov8_param = CVI_TDL_GetDetectionAlgoParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION);
    yolov8_param.cls = MODEL_CLASS_CNT;
    ret = CVI_TDL_SetDetectionAlgoParam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, yolov8_param);
    if (ret != CVI_SUCCESS) {
      throw std::runtime_error("Can not set yolov8 algorithm parameters" + std::to_string(ret));
    }
    CVI_TDL_SetModelThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, MODEL_THRESH);
    CVI_TDL_SetModelNmsThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, MODEL_NMS_THRESH);
    ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, modelFilePath);
    if (ret != CVI_SUCCESS) {
        throw std::runtime_error("Open model failed with error code: " + std::to_string(ret));
    }
}




int detect_marker(cv::Mat *bgr){
    cv::Mat grayImage, markerImage, thresh;
    cvtColor(*bgr, grayImage, COLOR_BGR2GRAY);
    cv::adaptiveThreshold(grayImage, thresh, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 11, 7);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    cv::findContours(thresh, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    

    for (size_t i = 0; i < contours.size(); ++i) {
        if (hierarchy[i][3] != -1)
            continue;

        double area = cv::contourArea(contours[i]);
        if (area < 100)
            continue; // Skip small noise

        vector<Point> approx;
        cv::approxPolyDP(contours[i], approx, cv::arcLength(contours[i], true) * 0.02, true);

        // Search quadrilaterals
        if (approx.size() == 4 && cv::isContourConvex(approx)) {
            Rect bound = cv::boundingRect(approx);
            float aspectRatio = (float)bound.width / bound.height;
            if (aspectRatio < 0.8 || aspectRatio > 1.2)
                continue;

            // // Draw corners
            // for (int i = 0; i < 4; i++) {
            //     cv::circle(bgr, approx[i], 5, Scalar(128), FILLED);
            // }

            // printf("Corners:\n");
            // for (const Point &pt : approx) {
            //     printf("x=%d; y=%d\n", pt.x, pt.y);
            // }

            std::vector<cv::Point2f> srcPts;
            for (const cv::Point &pt : approx)
                srcPts.push_back(pt);
            
            cv::Point2f center(0, 0);
            for (const cv::Point2f &pt : srcPts)
                center += pt;
            center *= (1.0 / srcPts.size());

            // Sort: top-left, top-right, bottom-right, bottom-left
            std::vector<cv::Point2f> orderedPts(4);
            for (const cv::Point2f &pt : srcPts) {
                if (pt.x < center.x && pt.y < center.y)
                    orderedPts[0] = pt; // Top-left
                else if (pt.x > center.x && pt.y < center.y)
                    orderedPts[1] = pt; // Top-right
                else if (pt.x > center.x && pt.y > center.y)
                    orderedPts[2] = pt; // Bottom-right
                else if (pt.x < center.x && pt.y > center.y)
                    orderedPts[3] = pt; // Bottom-left
            }

            cv::Mat warpMat = cv::getPerspectiveTransform(orderedPts, dstPts);
            cv::warpPerspective(grayImage, markerImage, warpMat, cv::Size(markerSize_, markerSize_));
            
            threshold(markerImage, thresh, 127, 255, THRESH_BINARY);

            // resize(thresh, thresh, Size(70, 70)); // , 0, 0, cv::INTER_NEAREST
            
            cv::Mat bitsMat = extractMarkerBits(thresh);
            
            int id = -1, rotation = -1;
            bool valid = dictionary.identify(bitsMat, id, rotation, false);
            // printf("Valid: %d; ID: %d; Rot: %d\n", valid, id, rotation);

            return id;

        }
    }

    return -1;
}


int main(int argc, char *argv[]) {
    signal(SIGINT, interrupt_handler);
    // Camera init
    cv::VideoCapture cap;
    test = MJPEGWriter(7777);

    dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);

    cap.set(cv::CAP_PROP_FRAME_WIDTH, VIDEO_RECORD_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, VIDEO_RECORD_FRAME_HEIGHT);
    cap.open(0);

    cv::Mat bgr;
    // "Warmup" camera
    for (int i = 0; i < 5; i++) {
        cap >> bgr;
    }
    printf("Warmup finished\n");

    test.write(bgr);
    test.start();

    initModel(argv[1]);
    
    while (!interrupted) {
        cv::Mat frame;
        std::pair<void *, void *> imagePtrs = cap.capture(frame);
        void *image_ptr = imagePtrs.first;

        if (image_ptr != nullptr) {
            VIDEO_FRAME_INFO_S *frameInfo = reinterpret_cast<VIDEO_FRAME_INFO_S *>(image_ptr);
            cvtdl_object_t obj_meta = {0};
            auto begin = std::chrono::steady_clock::now();
            CVI_TDL_Detection(tdl_handle, frameInfo, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, &obj_meta);
            cap.releaseImagePtr();
            image_ptr = nullptr;
            // printf("Detected: %d\n", obj_meta.size);
            
            for (uint32_t i = 0; i < obj_meta.size; i++)
            {
                cv::Mat res;
                cv::Rect r = cv::Rect(obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1, obj_meta.info[i].bbox.x2 - obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y2 - obj_meta.info[i].bbox.y1);
                res = frame(r).clone();
                // printf("res size: %dx%d, channels: %d, Empty: %d, Continuous: %d\n", res.cols, res.rows, bgr.channels(), res.empty(), res.isContinuous());
                // resize(res, normalized, Size(20, 20));
                // printf("Shape: %dx%d\n", res.cols, res.rows);
                // if (!res.empty()){
                //     resize(res, res, Size(100, 100), 0, 0, cv::INTER_NEAREST);
                // } else {
                //     printf("Empty\n");
                // }
                int id = detect_marker(&res);
                // printf("ID: %d\n", id);
                // test.write(frame);

            }
            auto end = std::chrono::steady_clock::now();
            double fps = 1 / std::chrono::duration<double>(end - begin).count();
            printf("Yolo + CV Detector FPS: %lf\n", fps);
            
        }

        /*

        0. Detect Marker with Yolo
        1. Crop marker from camera img
        2. * Maybe resize to some small square resolution (for example, 100x100 px)
        3. Execute cv detection pipeline: detect_marker()
        */

        // auto begin = std::chrono::steady_clock::now();
        // int id = detect_marker(&bgr);
        // auto end = std::chrono::steady_clock::now();
        // double fps = 1 / std::chrono::duration<double>(end - begin).count();
        // printf("Detector FPS: %lf\n", fps);
        // printf("Detected ID: %d\n", id);

        
    }

    cap.release();

    return 0;
}