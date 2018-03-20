cmake_minimum_required(VERSION 3.10)
project(ceplact)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(/usr/local/include)
link_directories(/usr/local/lib)

add_executable(main src/main.cpp)

target_link_libraries(main opencv_core opencv_imgproc opencv_imgcodecs opencv_calib3d opencv_features2d opencv_flann opencv_highgui opencv_ml opencv_objdetect opencv_photo opencv_stitching opencv_superres opencv_video opencv_videostab)

target_include_directories(main PUBLIC src)


include(Testing)
include(AddTests)
