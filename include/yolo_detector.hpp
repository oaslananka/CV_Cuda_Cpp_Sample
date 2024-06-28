#ifndef YOLO_DETECTOR_HPP
#define YOLO_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>
#include <vector>
#include <string>

/**
 * @brief The YOLODetector class is responsible for detecting objects using the YOLO (You Only Look Once) algorithm.
 */
class YOLODetector
{
public:
    /**
     * @brief Constructs a YOLODetector object with the specified configuration, weights, and classes file paths.
     * @param configPath The path to the YOLO configuration file.
     * @param weightsPath The path to the YOLO weights file.
     * @param classesFile The path to the file containing the class names.
     */
    YOLODetector(const std::string &configPath, const std::string &weightsPath, const std::string &classesFile);

    /**
     * @brief Detects objects in the given frame using the YOLO algorithm.
     * @param frame The input frame to perform object detection on.
     */
    void detect(cv::Mat &frame);

private:
    /**
     * @brief Draws a bounding box around the detected object on the given frame.
     * @param classId The ID of the detected object class.
     * @param conf The confidence score of the detection.
     * @param left The x-coordinate of the top-left corner of the bounding box.
     * @param top The y-coordinate of the top-left corner of the bounding box.
     * @param right The x-coordinate of the bottom-right corner of the bounding box.
     * @param bottom The y-coordinate of the bottom-right corner of the bounding box.
     * @param frame The frame to draw the bounding box on.
     */
    void drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame);

    cv::dnn::Net net;
    std::vector<std::string> classes;
};

#endif // YOLO_DETECTOR_HPP
