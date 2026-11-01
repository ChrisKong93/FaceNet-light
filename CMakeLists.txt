cmake_minimum_required(VERSION 3.5)
project(main)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package( OpenCV REQUIRED)

#set(CMAKE_BUILD_TYPE Debug)
AUX_SOURCE_DIRECTORY(./src DIR_SRCS)
add_executable( main ${DIR_SRCS})
target_link_libraries( main ${OpenCV_LIBS} )
