cmake_minimum_required(VERSION 3.0)
project(appimage)

file(COPY ${CMAKE_SOURCE_DIR}/pictures/numbers.jpg DESTINATION ${CMAKE_BINARY_DIR})
# Set the path to the OpenCV build directory
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp_lib/build)

# Set the path to the Eigen directory
set(EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../cpp_lib/eigen-3.4.0)

find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS} )

# Include Eigen headers
include_directories(${EIGEN_DIR})

add_executable(appimage encoder.cpp mnist2.cpp "neural_network_copy1 copy.cpp")

# Link against OpenCV libraries
target_link_libraries(appimage ${OpenCV_LIBS})

set_target_properties(appimage PROPERTIES 
    INSTALL_RPATH "${OpenCV_DIR}/lib"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

install(
    TARGETS appimage
    RUNTIME DESTINATION bin
)

install(
    FILES ${CMAKE_SOURCE_DIR}/pictures/numbers.jpg
    DESTINATION bin
)

install(
    FILES qtquickapp.desktop
    DESTINATION share/applications/
)

install(
    FILES qtquickapp.png
    DESTINATION share/icons/hicolor/128x128/apps
)