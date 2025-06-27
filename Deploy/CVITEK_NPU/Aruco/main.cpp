#include "MJPEGWriter.h"
#include <chrono>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <signal.h>
#include <stdio.h>

#define DEVICE_PATH 0
#define VIDEO_RECORD_FRAME_WIDTH 200
#define VIDEO_RECORD_FRAME_HEIGHT 200

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


int main() {
    signal(SIGINT, interrupt_handler);
    // Camera init
    cv::VideoCapture cap;
    MJPEGWriter test(7777);

    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
    cv::Mat grayImage, markerImage, thresh;

    float markerSize_ = 100; // pixels
    std::vector<cv::Point2f> dstPts = {
        cv::Point2f(0, 0),
        cv::Point2f(markerSize_ - 1, 0),
        cv::Point2f(markerSize_ - 1, markerSize_ - 1),
        cv::Point2f(0, markerSize_ - 1)
    };

    cap.set(cv::CAP_PROP_FRAME_WIDTH, VIDEO_RECORD_FRAME_WIDTH);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, VIDEO_RECORD_FRAME_HEIGHT);
    cap.open(0);

    cv::Mat bgr;
    // "Warmup" camera
    for (int i = 0; i < 5; i++) {
        cap >> bgr;
    }

    test.write(bgr);
    test.start();

    printf("Warmup finished\n");
    

    while (!interrupted) {
        cap >> bgr;

        if (bgr.empty()) {
            printf("Frame empty\n");
            interrupted = 1;
            continue;
        }

        cvtColor(bgr, grayImage, COLOR_BGR2GRAY);
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
                for (int i = 0; i < 4; i++) {
                    cv::circle(bgr, approx[i], 5, Scalar(128), FILLED);
                }

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

                // resize(thresh, thresh, Size(70, 70), 0, 0, cv::INTER_NEAREST);
                cv::Mat bitsMat = extractMarkerBits(thresh);
                
                int id = -1, rotation = -1;
                bool valid = dictionary.identify(bitsMat, id, rotation, false);

                printf("ID: %d\n", id);

            }
        }

        test.write(bgr);
        bgr.release();
    }

    cap.release();

    return 0;
}