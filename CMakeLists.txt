cmake_minimum_required(VERSION 3.0)
project(opencvtest)

# Set the path to the OpenCV build directory
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp_lib)

find_package(OpenCV REQUIRED)
# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencvtest opencvtest.cpp)

# Link against OpenCV libraries
target_link_libraries(opencvtest ${OpenCV_LIBS})
