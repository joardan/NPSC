# CMake generated Testfile for 
# Source directory: /media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/flann
# Build directory: /media/joardan/Harddisk/cpp_lib/build/modules/flann
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_flann "/media/joardan/Harddisk/cpp_lib/build/bin/opencv_test_flann" "--gtest_output=xml:opencv_test_flann.xml")
set_tests_properties(opencv_test_flann PROPERTIES  LABELS "Main;opencv_flann;Accuracy" WORKING_DIRECTORY "/media/joardan/Harddisk/cpp_lib/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVUtils.cmake;1795;add_test;/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/media/joardan/Harddisk/cpp_lib/opencv-4.x/cmake/OpenCVModule.cmake;1133;ocv_add_accuracy_tests;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/flann/CMakeLists.txt;2;ocv_define_module;/media/joardan/Harddisk/cpp_lib/opencv-4.x/modules/flann/CMakeLists.txt;0;")
