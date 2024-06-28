#include "yolo_detector.hpp"

/**
 * @brief The main function of the program.
 *
 * @param argc The number of command-line arguments.
 * @param argv An array of command-line arguments.
 * @return int The exit status of the program.
 */
int main(int argc, char **argv)
{
    const std::string keys =
        "{help h usage ? | | Usage examples: ./yolo_video --video=video.mp4 --device=0 }"
        "{video v        | | Path to the video file. If not specified, camera input will be used. }"
        "{device d       | 0 | Device index for camera input (default is 0). }"
        "{weights        | yolo.weights | Path to the YOLO weights file. }"
        "{config         | yolo.cfg | Path to the YOLO config file. }"
        "{classes        | coco.names | Path to the file with class names. }";

    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Use this script to run YOLO object detection on video input");
    if (parser.has("help"))
    {
        parser.printMessage();
        return 0;
    }
    // int deviceId = 0;
    // std::string videoPath = "drone.mp4";
    // std::string weightsPath = "data/yolo.weights";
    // std::string configPath = "data/yolo.cfg";
    // std::string classesFile = "data/coco.names";

    std::string videoPath = parser.get<std::string>("video");
    int deviceId = parser.get<int>("device");
    std::string weightsPath = parser.get<std::string>("weights");
    std::string configPath = parser.get<std::string>("config");
    std::string classesFile = parser.get<std::string>("classes");

    YOLODetector detector(configPath, weightsPath, classesFile);

    cv::VideoCapture cap;
    if (videoPath.empty())
    {
        cap.open(deviceId, cv::CAP_ANY);
    }
    else
    {
        cap.open(videoPath, cv::CAP_FFMPEG);
    }

    if (!cap.isOpened())
    {
        std::cerr << "Error opening video stream or file" << std::endl;
        return -1;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    double delay = 1000 / fps;

    cv::Mat frame;
    while (cap.read(frame))
    {
        detector.detect(frame);

        cv::imshow("Frame", frame);
        if (cv::waitKey(delay) >= 0)
            break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
