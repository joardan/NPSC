# CMake generated Testfile for 
# Source directory: /media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/dnn
# Build directory: /media/joardan/Harddisk/cpp_lib/build/modules/dnn
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_dnn "/media/joardan/Harddisk/cpp_lib/build/bin/opencv_test_dnn" "--gtest_output=xml:opencv_test_dnn.xml")
set_tests_properties(opencv_test_dnn PROPERTIES  LABELS "Main;opencv_dnn;Accuracy" WORKING_DIRECTORY "/media/joardan/Harddisk/cpp_lib/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVUtils.cmake;1795;add_test;/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/dnn/CMakeLists.txt;247;ocv_add_accuracy_tests;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/dnn/CMakeLists.txt;0;")
add_test(opencv_perf_dnn "/media/joardan/Harddisk/cpp_lib/build/bin/opencv_perf_dnn" "--gtest_output=xml:opencv_perf_dnn.xml")
set_tests_properties(opencv_perf_dnn PROPERTIES  LABELS "Main;opencv_dnn;Performance" WORKING_DIRECTORY "/media/joardan/Harddisk/cpp_lib/build/test-reports/performance" _BACKTRACE_TRIPLES "/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVUtils.cmake;1795;add_test;/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVModule.cmake;1274;ocv_add_test_from_target;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/dnn/CMakeLists.txt;258;ocv_add_perf_tests;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/dnn/CMakeLists.txt;0;")
add_test(opencv_sanity_dnn "/media/joardan/Harddisk/cpp_lib/build/bin/opencv_perf_dnn" "--gtest_output=xml:opencv_perf_dnn.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_dnn PROPERTIES  LABELS "Main;opencv_dnn;Sanity" WORKING_DIRECTORY "/media/joardan/Harddisk/cpp_lib/build/test-reports/sanity" _BACKTRACE_TRIPLES "/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVUtils.cmake;1795;add_test;/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVModule.cmake;1275;ocv_add_test_from_target;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/dnn/CMakeLists.txt;258;ocv_add_perf_tests;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/dnn/CMakeLists.txt;0;")
