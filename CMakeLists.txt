# Set the minimum required version of CMake
cmake_minimum_required(VERSION 3.10)

# Set the project name
project(CV_Cuda_Cpp_Sample)

# Set the CMake policy
cmake_policy(SET CMP0146 OLD)

# Find the CUDA package
find_package(CUDA REQUIRED)

# Find the OpenCV package
find_package(OpenCV REQUIRED)

# Include the directories for OpenCV and CUDA
include_directories(${OpenCV_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} include)

# Get the list of source files
file(GLOB SOURCES "src/*.cpp")

# Set the output directory for the executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_SOURCE_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_SOURCE_DIR})

# Create the executable target
add_executable(CV_Cuda_Cpp_Sample ${SOURCES})

# Link the necessary libraries to the executable
target_link_libraries(CV_Cuda_Cpp_Sample ${OpenCV_LIBS} ${CUDA_LIBRARIES} opencv_core opencv_imgproc opencv_highgui opencv_cudaimgproc opencv_video)
