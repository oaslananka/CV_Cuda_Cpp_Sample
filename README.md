# People Detection from Drone Footage
![](assests/output.gif)


This is a sample project demonstrating how to use OpenCV and CUDA in C++ for detecting people in drone footage with YOLO. The project aims to be simple and understandable for those who want to learn how to use OpenCV and CUDA in C++.

## OpenCV Sources

- [OpenCV](https://github.com/opencv/opencv)
- [OpenCV Contrib](https://github.com/opencv/opencv_contrib)

## Sample Videos

Sample videos are taken from the link below:

- [Kaggle Drone Videos Dataset](https://www.kaggle.com/datasets/kmader/drone-videos/data)

## Build Instructions

Compile the project in release mode with CMake. The executable output will be placed directly in the project root folder.

### Prerequisites

- CMake
- OpenCV
- CUDA
- C++ Compiler

### Steps

1. Clone the repository:
   ```sh
   git clone https://github.com/your_username/PeopleDetectFromDrone.git
   cd PeopleDetectFromDrone
   ```
2. Create a build directory and navigate to it:
   ```sh
   mkdir build
   cd build
   ```
3. Run CMake to configure the project:
    ```sh
    cmake ..
    ```
4. Compile the project:
    ```sh
    cmake --build . --config Release
    ```

## Usage
- For argument help:
    ```sh
    ./OpenCV_Cuda_Cpp_Sample -h
    ```


- You can try it with any video by providing the arguments to the executable. Sample usage code is below:
    ```sh
    ./OpenCV_Cuda_Cpp_Sample -video=video.mp4 -device=0 -weights=yolo.weights -config=yolo.cfg -classes=coco.names
    ```
