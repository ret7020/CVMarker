# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

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
CMAKE_COMMAND = /home/stephan/.local/lib/python3.11/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/stephan/.local/lib/python3.11/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/build

# Include any dependencies generated for this target.
include CMakeFiles/stream_yolov8.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/stream_yolov8.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/stream_yolov8.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/stream_yolov8.dir/flags.make

CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o: CMakeFiles/stream_yolov8.dir/flags.make
CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/MJPEGWriter.cpp
CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o: CMakeFiles/stream_yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o"
	/home/stephan/Downloads/LicheeRV_CrossCompilers/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o -MF CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o.d -o CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o -c /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/MJPEGWriter.cpp

CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.i"
	/home/stephan/Downloads/LicheeRV_CrossCompilers/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/MJPEGWriter.cpp > CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.i

CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.s"
	/home/stephan/Downloads/LicheeRV_CrossCompilers/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/MJPEGWriter.cpp -o CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.s

CMakeFiles/stream_yolov8.dir/main.c.o: CMakeFiles/stream_yolov8.dir/flags.make
CMakeFiles/stream_yolov8.dir/main.c.o: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/main.c
CMakeFiles/stream_yolov8.dir/main.c.o: CMakeFiles/stream_yolov8.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/stream_yolov8.dir/main.c.o"
	/home/stephan/Downloads/LicheeRV_CrossCompilers/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-g++ $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/stream_yolov8.dir/main.c.o -MF CMakeFiles/stream_yolov8.dir/main.c.o.d -o CMakeFiles/stream_yolov8.dir/main.c.o -c /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/main.c

CMakeFiles/stream_yolov8.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing C source to CMakeFiles/stream_yolov8.dir/main.c.i"
	/home/stephan/Downloads/LicheeRV_CrossCompilers/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-g++ $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/main.c > CMakeFiles/stream_yolov8.dir/main.c.i

CMakeFiles/stream_yolov8.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling C source to assembly CMakeFiles/stream_yolov8.dir/main.c.s"
	/home/stephan/Downloads/LicheeRV_CrossCompilers/gcc/riscv64-linux-musl-x86_64/bin/riscv64-unknown-linux-musl-g++ $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/main.c -o CMakeFiles/stream_yolov8.dir/main.c.s

# Object files for target stream_yolov8
stream_yolov8_OBJECTS = \
"CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o" \
"CMakeFiles/stream_yolov8.dir/main.c.o"

# External object files for target stream_yolov8
stream_yolov8_EXTERNAL_OBJECTS =

bin/stream_yolov8: CMakeFiles/stream_yolov8.dir/MJPEGWriter.cpp.o
bin/stream_yolov8: CMakeFiles/stream_yolov8.dir/main.c.o
bin/stream_yolov8: CMakeFiles/stream_yolov8.dir/build.make
bin/stream_yolov8: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/libs/opencv-mobile-4.10.0-licheerv-nano/lib/libopencv_core.a
bin/stream_yolov8: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/libs/opencv-mobile-4.10.0-licheerv-nano/lib/libopencv_features2d.a
bin/stream_yolov8: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/libs/opencv-mobile-4.10.0-licheerv-nano/lib/libopencv_highgui.a
bin/stream_yolov8: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/libs/opencv-mobile-4.10.0-licheerv-nano/lib/libopencv_imgproc.a
bin/stream_yolov8: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/libs/opencv-mobile-4.10.0-licheerv-nano/lib/libopencv_photo.a
bin/stream_yolov8: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/libs/opencv-mobile-4.10.0-licheerv-nano/lib/libopencv_video.a
bin/stream_yolov8: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/libs/opencv-mobile-4.10.0-licheerv-nano/lib/libopencv_imgproc.a
bin/stream_yolov8: /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/libs/opencv-mobile-4.10.0-licheerv-nano/lib/libopencv_core.a
bin/stream_yolov8: CMakeFiles/stream_yolov8.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable bin/stream_yolov8"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/stream_yolov8.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/stream_yolov8.dir/build: bin/stream_yolov8
.PHONY : CMakeFiles/stream_yolov8.dir/build

CMakeFiles/stream_yolov8.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/stream_yolov8.dir/cmake_clean.cmake
.PHONY : CMakeFiles/stream_yolov8.dir/clean

CMakeFiles/stream_yolov8.dir/depend:
	cd /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/build /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/build /home/stephan/Progs/CVMarker/Deploy/CVITEK_NPU/4_POINTS/build/CMakeFiles/stream_yolov8.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/stream_yolov8.dir/depend

