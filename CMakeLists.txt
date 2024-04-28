cmake_minimum_required(VERSION 3.25)
project(untitled)

set(CMAKE_CXX_STANDARD 17)

add_executable(untitled tensor.cpp transformer.cpp unet.cpp unet.cpp load.cpp load.cpp load.h load.cpp main.cpp)
