/*
Init sequence:
Init camera
Init yolo
Get first detection and correct marker estimation
Init mavlink
Warmup mavlink with correct vpe and correct target position
Put into offboard and ARM
While true sending vpe; target loop
*/

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
#include <time.h>
#include <unistd.h>

#include "config.h"
#include "yolo.h"

// Globals
cv::VideoCapture cap;
cvitdl_handle_t tdl_handle = nullptr;
volatile sig_atomic_t interrupted = 0;

void clean_up() {
    cap.release();
    if (tdl_handle != nullptr) {
        CVI_TDL_DestroyHandle(tdl_handle);
        tdl_handle = nullptr;
    }
}

void interrupt_handler(int signum) {
    printf("[MAIN] Signal: %d\n", signum);
    interrupted = 1;
}

int detect_marker(cv::Vec3d *translation_result, cv::Vec3f *orientation_result) {
    /*
    0  - inference ok; marker detected
    -1 - inference not ok; frame is null
    -2 - inference ok; marker not detected (not all corners)
    */
    cv::Mat bgr;
    cvtdl_object_t obj_meta = {0};
    cv::Point2f coords[4];
    float max_scores[4] = {0};

    std::pair<void *, void *> imagePtrs = cap.capture(bgr);
    void *image_ptr = imagePtrs.first;
    VIDEO_FRAME_INFO_S *frameInfo = reinterpret_cast<VIDEO_FRAME_INFO_S *>(image_ptr);

    // Check frame read ok and inference
    if (image_ptr != nullptr) {
        CVI_TDL_YOLOV8_Detection(tdl_handle, frameInfo, &obj_meta);
        cap.releaseImagePtr();
        image_ptr = nullptr;
    } else { // prevent seg fault
        interrupted = 1;
        return -1;
    }

    for (uint32_t i = 0; i < obj_meta.size; i++) {
        int cls = obj_meta.info[i].classes;

        float score = obj_meta.info[i].bbox.score;
        if (score > max_scores[cls]) {
            max_scores[cls] = score;

            float c_x = (obj_meta.info[i].bbox.x2 + obj_meta.info[i].bbox.x1) * 0.5f;
            float c_y = (obj_meta.info[i].bbox.y2 + obj_meta.info[i].bbox.y1) * 0.5f;
            coords[cls] = cv::Point2f(c_x, c_y);
        }
    }

    int detected_corners = 0;
    for (int i = 0; i < 4; i++) {
        if (max_scores[i] > 0) {
            detected_corners++;
        }
    }

    if (detected_corners != 4) {
// Debug frame save (with invalid detection)
#ifdef DEBUG_SAVE_ON_FAIL
        for (int i = 0; i < 4; i++) {
            if (max_scores[i] > 0) {
                cv::Point2f pt = coords[i];
                cv::circle(bgr, pt, 5, color_map[i], -1);
                cv::putText(bgr, std::to_string(i), pt + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            color_map[i], 1);
            }
        }

        char filename[64];
        sprintf(filename, "./failed_markers/%d.jpg", (int)time(NULL));
        cv::imwrite(filename, bgr); // WARN this is time consuming operation that will affect latency
#endif

        return -2;
    }

    std::vector<cv::Point2f> image_points = {
        coords[2], // Bottom-left
        coords[3], // Bottom-right
        coords[1], // Top-right
        coords[0]  // Top-left
    };

    cv::Mat rvec, tvec;
    bool success = cv::solvePnP(object_points, image_points, camera_matrix, dist_coeffs, rvec, tvec, false,
                                cv::SOLVEPNP_IPPE); // object_points - ref to config.h

    cv::Vec3d position(tvec);
    // Position in meters
    position[0] *= -0.001;
    position[1] *= -0.001;
    position[2] *= -0.001;

    cv::Mat R_cam_to_marker;
    cv::Rodrigues(rvec, R_cam_to_marker);
    cv::Mat R_marker_to_cam = R_cam_to_marker.t();
    cv::Vec3f orientation = rot_to_euler(R_marker_to_cam);
    *translation_result = position;
    *orientation_result = orientation;
    return 0;
}

int main(int argc, char *argv[]) {

    signal(SIGINT, interrupt_handler); // Correct program stop

    CVI_S32 ret;
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, FRAME_SIZE);
    cap.set(cv::CAP_PROP_FRAME_WIDTH, FRAME_SIZE);

    cap.open(0);

    ret = CVI_TDL_CreateHandle(&tdl_handle);
    if (ret != CVI_SUCCESS) {
        printf("[MAIN] Create tdl handle failed with %#x!\n", ret);
        return ret;
    }

    ret = SET_YOLO_Params(tdl_handle);
    ret = CVI_TDL_OpenModel(tdl_handle, CVI_TDL_SUPPORTED_MODEL_YOLOV8_DETECTION, argv[1]);

    if (ret != CVI_SUCCESS) {
        printf("[MAIN] Open model failed with %#x!\n", ret);
        return ret;
    }
    printf("[MAIN] Detector ready\n");

    cv::Vec3d marker_position;
    cv::Vec3f marker_orientation;

    // First detection
    while (!interrupted) {
        int status = detect_marker(&marker_position, &marker_orientation);
        if (status < 0)
            printf("[MAIN] Marker lost\n");
        else
            printf("[MAIN] Marker detected; VPE: x=%lf; y=%lf; z=%lf\n", marker_position[0], marker_position[1],
                   marker_position[2]);
    }

    printf("Exit...\n");
    clean_up();

    return 0;
}