# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /media/joardan/Harddisk/Project/NPSC

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /media/joardan/Harddisk/Project/NPSC/build

# Include any dependencies generated for this target.
include CMakeFiles/appimage.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/appimage.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/appimage.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/appimage.dir/flags.make

CMakeFiles/appimage.dir/classify.cpp.o: CMakeFiles/appimage.dir/flags.make
CMakeFiles/appimage.dir/classify.cpp.o: ../classify.cpp
CMakeFiles/appimage.dir/classify.cpp.o: CMakeFiles/appimage.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/media/joardan/Harddisk/Project/NPSC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/appimage.dir/classify.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/appimage.dir/classify.cpp.o -MF CMakeFiles/appimage.dir/classify.cpp.o.d -o CMakeFiles/appimage.dir/classify.cpp.o -c /media/joardan/Harddisk/Project/NPSC/classify.cpp

CMakeFiles/appimage.dir/classify.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/appimage.dir/classify.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /media/joardan/Harddisk/Project/NPSC/classify.cpp > CMakeFiles/appimage.dir/classify.cpp.i

CMakeFiles/appimage.dir/classify.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/appimage.dir/classify.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /media/joardan/Harddisk/Project/NPSC/classify.cpp -o CMakeFiles/appimage.dir/classify.cpp.s

# Object files for target appimage
appimage_OBJECTS = \
"CMakeFiles/appimage.dir/classify.cpp.o"

# External object files for target appimage
appimage_EXTERNAL_OBJECTS =

appimage: CMakeFiles/appimage.dir/classify.cpp.o
appimage: CMakeFiles/appimage.dir/build.make
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_gapi.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_highgui.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_ml.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_objdetect.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_photo.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_stitching.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_video.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_videoio.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_imgcodecs.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_dnn.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_calib3d.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_features2d.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_flann.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_imgproc.so.4.9.0
appimage: /media/joardan/Harddisk/cpp_lib/build/lib/libopencv_core.so.4.9.0
appimage: CMakeFiles/appimage.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/media/joardan/Harddisk/Project/NPSC/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable appimage"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/appimage.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/appimage.dir/build: appimage
.PHONY : CMakeFiles/appimage.dir/build

CMakeFiles/appimage.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/appimage.dir/cmake_clean.cmake
.PHONY : CMakeFiles/appimage.dir/clean

CMakeFiles/appimage.dir/depend:
	cd /media/joardan/Harddisk/Project/NPSC/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /media/joardan/Harddisk/Project/NPSC /media/joardan/Harddisk/Project/NPSC /media/joardan/Harddisk/Project/NPSC/build /media/joardan/Harddisk/Project/NPSC/build /media/joardan/Harddisk/Project/NPSC/build/CMakeFiles/appimage.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/appimage.dir/depend

