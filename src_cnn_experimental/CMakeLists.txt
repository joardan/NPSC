cmake_minimum_required(VERSION 3.0)
project(cnn)

# Set the path to the OpenCV build directory
set(OpenCV_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp_lib/build)

# Set the path to the Eigen directory
set(EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../cpp_lib/eigen-3.4.0)

find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS} )

# Include Eigen headers
include_directories(${EIGEN_DIR})

# Include CNN headers
include_directories(../include_cnn)

# Build Test CNN
add_executable(test multi_numbers_processing.cpp encoder.cpp mnist2.cpp conv_layer_unoptimised.cpp conv_max_pool.cpp conv_fc_layer.cpp conv_function.cpp conv_main_test.cpp)
# Build Run CNN
add_executable(run multi_numbers_processing.cpp encoder.cpp mnist2.cpp conv_layer_unoptimised.cpp conv_max_pool.cpp conv_fc_layer.cpp conv_function.cpp conv_main_run.cpp)
# Build Train CNN
add_executable(train multi_numbers_processing.cpp encoder.cpp mnist2.cpp conv_layer_unoptimised.cpp conv_max_pool.cpp conv_fc_layer.cpp conv_function.cpp conv_main_train.cpp)

# Link against OpenCV libraries
target_link_libraries(test ${OpenCV_LIBS})
target_link_libraries(train ${OpenCV_LIBS})
target_link_libraries(run ${OpenCV_LIBS})

install(
    TARGETS test run train
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