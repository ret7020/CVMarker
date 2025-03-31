#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <chrono>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <signal.h>
#include "core/cvi_tdl_types_mem_internal.h"
#include "core/utils/vpss_helper.h"
#include "cvi_tdl.h"
#include "cvi_tdl_media.h"
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "MJPEGWriter.h"

#define MODEL_SCALE 0.0039216
#define MODEL_MEAN 0.0
#define MODEL_CLASS_CNT 4
#define MODEL_THRESH 0.2
#define MODEL_NMS_THRESH 0.2
#define BLUE_MAT cv::Scalar(255, 0, 0)
#define RED_MAT cv::Scalar(0, 0, 255)

double square_size = 54.0;

cv::Scalar color_map[4] = {cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(100, 100, 100)};

// TODO replace
cv::Mat camera_matrix = (cv::Mat_<double>(3, 3) << 727.97277723, 0, 308.83841529,
                         0, 723.12158831, 270.40274403,
                         0, 0, 1);
cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << -0.71612235, 0.61866812, -0.03516522, 0.00746986, -0.35457296);

std::vector<cv::Point3f> object_points = {
    cv::Point3f(-square_size / 2, -square_size / 2, 0),
    cv::Point3f(square_size / 2, -square_size / 2, 0),
    cv::Point3f(square_size / 2, square_size / 2, 0),
    cv::Point3f(-square_size / 2, square_size / 2, 0)};

volatile uint8_t interrupted = 0;

void interrupt_handler(int signum)
{
    printf("Signal: %d\n", signum);
    interrupted = 1;
}

CVI_S32 init_param(const cvitdl_handle_t tdl_handle)
{
    // setup preprocess
    YoloPreParam preprocess_cfg =
        CVI_TDL_Get_YOLO_Preparam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION);

    for (int i = 0; i < 3; i++)
    {
        printf("asign val %d \n", i);
        preprocess_cfg.factor[i] = MODEL_SCALE;
        preprocess_cfg.mean[i] = MODEL_MEAN;
    }
    preprocess_cfg.format = PIXEL_FORMAT_RGB_888_PLANAR;

    printf("setup yolov8 param \n");
    CVI_S32 ret = CVI_TDL_Set_YOLO_Preparam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION,
                                            preprocess_cfg);
    if (ret != CVI_SUCCESS)
    {
        printf("Can not set yolov8 preprocess parameters %#x\n", ret);
        return ret;
    }

    // setup yolo algorithm preprocess
    YoloAlgParam yolov8_param =
        CVI_TDL_Get_YOLO_Algparam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION);
    yolov8_param.cls = MODEL_CLASS_CNT;

    printf("setup yolov8 algorithm param \n");
    ret =
        CVI_TDL_Set_YOLO_Algparam(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, yolov8_param);
    if (ret != CVI_SUCCESS)
    {
        printf("Can not set yolov8 algorithm parameters %#x\n", ret);
        return ret;
    }

    // set theshold
    CVI_TDL_SetModelThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, MODEL_THRESH);
    CVI_TDL_SetModelNmsThreshold(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, MODEL_NMS_THRESH);

    printf("yolov8 algorithm parameters setup success!\n");
    return ret;
}

int main(int argc, char *argv[])
{

    signal(SIGINT, interrupt_handler);
    MJPEGWriter test(7777);

    cv::VideoCapture cap;
    cv::Mat bgr;

    cap.open(0);
    cap >> bgr;

    test.write(bgr);
    test.start();

    CVI_S32 ret;

    cvitdl_handle_t tdl_handle = NULL;
    ret = CVI_TDL_CreateHandle(&tdl_handle);
    if (ret != CVI_SUCCESS)
    {
        printf("Create tdl handle failed with %#x!\n", ret);
        return ret;
    }

    ret = init_param(tdl_handle);
    ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, argv[1]);

    if (ret != CVI_SUCCESS)
    {
        printf("open model failed with %#x!\n", ret);
        return ret;
    }


    while (!interrupted)
    {
        std::pair<void*, void*> imagePtrs = cap.capture(bgr);
        void* image_ptr = imagePtrs.first;

        VIDEO_FRAME_INFO_S *frameInfo = reinterpret_cast<VIDEO_FRAME_INFO_S*>(image_ptr);

        cvtdl_object_t obj_meta = {0};
        cv::Point2f coords[4];
        // std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        CVI_TDL_YOLOV8_Detection(tdl_handle, frameInfo, &obj_meta);
        cap.releaseImagePtr();
        image_ptr = nullptr;

        // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        // double fps = 1 / std::chrono::duration<double>(end - begin).count();
        // printf("\n\n----------\nDetection FPS: %lf\nDetected objects cnt: %d\n\nDetected objects:\n", fps, obj_meta.size);
        for (uint32_t i = 0; i < obj_meta.size; i++)
        {
            // printf("x1 = %lf, y1 = %lf, x2 = %lf, y2 = %lf, cls: %d, score: %lf\n", obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1, obj_meta.info[i].bbox.x2, obj_meta.info[i].bbox.y2, obj_meta.info[i].classes, obj_meta.info[i].bbox.score);
            cv::Rect r = cv::Rect(obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1, obj_meta.info[i].bbox.x2 - obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y2 - obj_meta.info[i].bbox.y1);

            int c_x = obj_meta.info[i].bbox.x1 + (obj_meta.info[i].bbox.x2 - obj_meta.info[i].bbox.x1) / 2;
            int c_y = obj_meta.info[i].bbox.y1 + (obj_meta.info[i].bbox.y2 - obj_meta.info[i].bbox.y1) / 2;
            cv::circle(bgr, cv::Point(c_x, c_y), 5, cv::Scalar(0, 0, 255), -1);
            cv::rectangle(bgr, r, color_map[obj_meta.info[i].classes], 1, 8, 0);
            coords[obj_meta.info[i].classes] = cv::Point2f(c_x, c_y);
            cv::putText(bgr,
                        std::to_string(obj_meta.info[i].classes),
                        cv::Point(obj_meta.info[i].bbox.x1, obj_meta.info[i].bbox.y1 - 5),
                        cv::FONT_HERSHEY_DUPLEX,
                        1.0,
                        color_map[obj_meta.info[i].classes],
                        1);
        }
        cv::line(bgr, coords[0], coords[1], (30, 100, 200), 5);
        cv::line(bgr, coords[1], coords[3], (30, 100, 200), 5);
        cv::line(bgr, coords[2], coords[3], (30, 100, 200), 5);
        cv::line(bgr, coords[0], coords[2], (30, 100, 200), 5);

        std::vector<cv::Point2f> image_points = {
            coords[2],  // Bottom-left
            coords[3], // Bottom-right
            coords[1], // Top-right
            coords[0]  // Top-left
        };
        cv::Mat rvec, tvec;
        bool success = cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec);
        double distance = cv::norm(tvec);
        printf("Dist: %lf mm.\n", distance);



        test.write(bgr);
        bgr.release();
    }

    printf("Stopping stream:\n");
    test.stop();
    cap.release();

    CVI_TDL_DestroyHandle(tdl_handle);

    return ret;
}
