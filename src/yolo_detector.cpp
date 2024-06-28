#include "yolo_detector.hpp"
#include <fstream>
#include <iostream>

/**
 * @brief Constructs a YOLODetector object.
 * 
 * This constructor initializes a YOLODetector object with the provided configuration, weights, and classes file paths.
 * It reads the classes from the classes file and stores them in the 'classes' vector.
 * It then loads the YOLO network from the darknet configuration and weights files.
 * Finally, it sets the preferable backend and target to CUDA for GPU acceleration.
 * 
 * @param configPath The path to the YOLO configuration file.
 * @param weightsPath The path to the YOLO weights file.
 * @param classesFile The path to the file containing the class names.
 */
YOLODetector::YOLODetector(const std::string &configPath, const std::string &weightsPath, const std::string &classesFile)
{
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (std::getline(ifs, line))
        classes.push_back(line);

    net = cv::dnn::readNetFromDarknet(configPath, weightsPath);
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
}

/**
 * @brief Performs object detection using YOLO algorithm on the given frame.
 * 
 * @param frame The input frame on which object detection is performed.
 */
void YOLODetector::detect(cv::Mat &frame)
{
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1 / 255.0, cv::Size(416, 416), cv::Scalar(0, 0, 0), true, false);
    net.setInput(blob);

    std::vector<cv::Mat> outs;
    net.forward(outs, net.getUnconnectedOutLayersNames());

    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;

    for (size_t i = 0; i < outs.size(); ++i)
    {
        float *data = (float *)outs[i].data;
        for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
        {
            cv::Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
            cv::Point classIdPoint;
            double confidence;
            cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
            if (confidence > 0.4)
            {
                int centerX = (int)(data[0] * frame.cols);
                int centerY = (int)(data[1] * frame.rows);
                int width = (int)(data[2] * frame.cols);
                int height = (int)(data[3] * frame.rows);
                int left = centerX - width / 2;
                int top = centerY - height / 2;

                classIds.push_back(classIdPoint.x);
                confidences.push_back((float)confidence);
                boxes.push_back(cv::Rect(left, top, width, height));
            }
        }
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.4f, indices);

    for (size_t i = 0; i < indices.size(); ++i)
    {
        int idx = indices[i];
        cv::Rect box = boxes[idx];
        drawPred(classIds[idx], confidences[idx], box.x, box.y, box.x + box.width, box.y + box.height, frame);
    }
}

/**
 * Draws a bounding box and label on the given frame based on the detection results.
 *
 * @param classId The ID of the detected class.
 * @param conf The confidence score of the detection.
 * @param left The left coordinate of the bounding box.
 * @param top The top coordinate of the bounding box.
 * @param right The right coordinate of the bounding box.
 * @param bottom The bottom coordinate of the bounding box.
 * @param frame The frame on which to draw the bounding box and label.
 */
void YOLODetector::drawPred(int classId, float conf, int left, int top, int right, int bottom, cv::Mat &frame)
{
    int centerX = (left + right) / 2;
    int centerY = (top + bottom) / 2;
    int radius = std::min(right - left, bottom - top);

    cv::circle(frame, cv::Point(centerX, centerY), radius, cv::Scalar(255, 0, 0), 3);

    std::string label = cv::format("%.2f", conf);
    if (!classes.empty() && classId < classes.size())
    {
        label = classes[classId] + ": " + label;
    }

    int baseLine;
    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.75, 1, &baseLine);
    int topText = std::max(top, labelSize.height);
    cv::rectangle(frame, cv::Point(left, topText - labelSize.height), cv::Point(left + labelSize.width, topText + baseLine), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::putText(frame, label, cv::Point(left, topText), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 1);
}