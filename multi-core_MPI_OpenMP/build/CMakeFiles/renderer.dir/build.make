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
CMAKE_SOURCE_DIR = /home/lor3n/Dev/parallel-raytracing

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/lor3n/Dev/parallel-raytracing/build

# Include any dependencies generated for this target.
include CMakeFiles/renderer.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/renderer.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/renderer.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/renderer.dir/flags.make

CMakeFiles/renderer.dir/src/Application.cpp.o: CMakeFiles/renderer.dir/flags.make
CMakeFiles/renderer.dir/src/Application.cpp.o: ../src/Application.cpp
CMakeFiles/renderer.dir/src/Application.cpp.o: CMakeFiles/renderer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lor3n/Dev/parallel-raytracing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/renderer.dir/src/Application.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/renderer.dir/src/Application.cpp.o -MF CMakeFiles/renderer.dir/src/Application.cpp.o.d -o CMakeFiles/renderer.dir/src/Application.cpp.o -c /home/lor3n/Dev/parallel-raytracing/src/Application.cpp

CMakeFiles/renderer.dir/src/Application.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/renderer.dir/src/Application.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lor3n/Dev/parallel-raytracing/src/Application.cpp > CMakeFiles/renderer.dir/src/Application.cpp.i

CMakeFiles/renderer.dir/src/Application.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/renderer.dir/src/Application.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lor3n/Dev/parallel-raytracing/src/Application.cpp -o CMakeFiles/renderer.dir/src/Application.cpp.s

CMakeFiles/renderer.dir/src/Renderer.cpp.o: CMakeFiles/renderer.dir/flags.make
CMakeFiles/renderer.dir/src/Renderer.cpp.o: ../src/Renderer.cpp
CMakeFiles/renderer.dir/src/Renderer.cpp.o: CMakeFiles/renderer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lor3n/Dev/parallel-raytracing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/renderer.dir/src/Renderer.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/renderer.dir/src/Renderer.cpp.o -MF CMakeFiles/renderer.dir/src/Renderer.cpp.o.d -o CMakeFiles/renderer.dir/src/Renderer.cpp.o -c /home/lor3n/Dev/parallel-raytracing/src/Renderer.cpp

CMakeFiles/renderer.dir/src/Renderer.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/renderer.dir/src/Renderer.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lor3n/Dev/parallel-raytracing/src/Renderer.cpp > CMakeFiles/renderer.dir/src/Renderer.cpp.i

CMakeFiles/renderer.dir/src/Renderer.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/renderer.dir/src/Renderer.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lor3n/Dev/parallel-raytracing/src/Renderer.cpp -o CMakeFiles/renderer.dir/src/Renderer.cpp.s

CMakeFiles/renderer.dir/src/Scene.cpp.o: CMakeFiles/renderer.dir/flags.make
CMakeFiles/renderer.dir/src/Scene.cpp.o: ../src/Scene.cpp
CMakeFiles/renderer.dir/src/Scene.cpp.o: CMakeFiles/renderer.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/lor3n/Dev/parallel-raytracing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/renderer.dir/src/Scene.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/renderer.dir/src/Scene.cpp.o -MF CMakeFiles/renderer.dir/src/Scene.cpp.o.d -o CMakeFiles/renderer.dir/src/Scene.cpp.o -c /home/lor3n/Dev/parallel-raytracing/src/Scene.cpp

CMakeFiles/renderer.dir/src/Scene.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/renderer.dir/src/Scene.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/lor3n/Dev/parallel-raytracing/src/Scene.cpp > CMakeFiles/renderer.dir/src/Scene.cpp.i

CMakeFiles/renderer.dir/src/Scene.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/renderer.dir/src/Scene.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/lor3n/Dev/parallel-raytracing/src/Scene.cpp -o CMakeFiles/renderer.dir/src/Scene.cpp.s

# Object files for target renderer
renderer_OBJECTS = \
"CMakeFiles/renderer.dir/src/Application.cpp.o" \
"CMakeFiles/renderer.dir/src/Renderer.cpp.o" \
"CMakeFiles/renderer.dir/src/Scene.cpp.o"

# External object files for target renderer
renderer_EXTERNAL_OBJECTS =

renderer: CMakeFiles/renderer.dir/src/Application.cpp.o
renderer: CMakeFiles/renderer.dir/src/Renderer.cpp.o
renderer: CMakeFiles/renderer.dir/src/Scene.cpp.o
renderer: CMakeFiles/renderer.dir/build.make
renderer: /usr/local/cuda-12.3/lib64/libcudart.so
renderer: /usr/local/lib/libmpi.so
renderer: CMakeFiles/renderer.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/lor3n/Dev/parallel-raytracing/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable renderer"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/renderer.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/renderer.dir/build: renderer
.PHONY : CMakeFiles/renderer.dir/build

CMakeFiles/renderer.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/renderer.dir/cmake_clean.cmake
.PHONY : CMakeFiles/renderer.dir/clean

CMakeFiles/renderer.dir/depend:
	cd /home/lor3n/Dev/parallel-raytracing/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/lor3n/Dev/parallel-raytracing /home/lor3n/Dev/parallel-raytracing /home/lor3n/Dev/parallel-raytracing/build /home/lor3n/Dev/parallel-raytracing/build /home/lor3n/Dev/parallel-raytracing/build/CMakeFiles/renderer.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/renderer.dir/depend

