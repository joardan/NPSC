cmake_minimum_required(VERSION 3.0)
project(opencvtest)

# Set the path to the OpenCV build directory
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/opencv4)

find_package(OpenCV REQUIRED)
# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(opencvtest opencvtest.cpp)

# Print the variables used in the Cmake, for debugging
message(STATUS "CMAKE_CURRENT_SOURCE_DIR = ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")

# Set RPATH, but only for the opencvtest executable. Seems to not affect anything.
# Probably wrong, change the RPATH of OpenCV build library instead?
set_target_properties(opencvtest PROPERTIES
    INSTALL_RPATH "$ORIGIN/../opencv4/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Link against OpenCV libraries
target_link_libraries(opencvtest ${OpenCV_LIBS})