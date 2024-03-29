# CMake generated Testfile for 
# Source directory: /media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/core
# Build directory: /media/joardan/Harddisk/cpp_lib/build/modules/core
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_core "/media/joardan/Harddisk/cpp_lib/build/bin/opencv_test_core" "--gtest_output=xml:opencv_test_core.xml")
set_tests_properties(opencv_test_core PROPERTIES  LABELS "Main;opencv_core;Accuracy" WORKING_DIRECTORY "/media/joardan/Harddisk/cpp_lib/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVUtils.cmake;1795;add_test;/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/core/CMakeLists.txt;197;ocv_add_accuracy_tests;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/core/CMakeLists.txt;0;")
add_test(opencv_perf_core "/media/joardan/Harddisk/cpp_lib/build/bin/opencv_perf_core" "--gtest_output=xml:opencv_perf_core.xml")
set_tests_properties(opencv_perf_core PROPERTIES  LABELS "Main;opencv_core;Performance" WORKING_DIRECTORY "/media/joardan/Harddisk/cpp_lib/build/test-reports/performance" _BACKTRACE_TRIPLES "/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVUtils.cmake;1795;add_test;/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVModule.cmake;1274;ocv_add_test_from_target;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/core/CMakeLists.txt;198;ocv_add_perf_tests;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/core/CMakeLists.txt;0;")
add_test(opencv_sanity_core "/media/joardan/Harddisk/cpp_lib/build/bin/opencv_perf_core" "--gtest_output=xml:opencv_perf_core.xml" "--perf_min_samples=1" "--perf_force_samples=1" "--perf_verify_sanity")
set_tests_properties(opencv_sanity_core PROPERTIES  LABELS "Main;opencv_core;Sanity" WORKING_DIRECTORY "/media/joardan/Harddisk/cpp_lib/build/test-reports/sanity" _BACKTRACE_TRIPLES "/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVUtils.cmake;1795;add_test;/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVModule.cmake;1275;ocv_add_test_from_target;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/core/CMakeLists.txt;198;ocv_add_perf_tests;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/core/CMakeLists.txt;0;")
