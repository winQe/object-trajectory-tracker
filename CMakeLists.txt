cmake_minimum_required(VERSION 3.0.2)
project(object_tracker)

find_package(OpenCV 4 REQUIRED) # manual specification of the OpenCVConfig.cmake path is necessary since otherwise, cmake assigns /opt/local a higher priority and ocv2.4 will be used
if(OpenCV_FOUND)
  MESSAGE ( STATUS "Found OpenCV: ${OpenCV_VERSION}" )
  MESSAGE ( STATUS "OpenCV_INCLUDE_DIRS= ${OpenCV_INCLUDE_DIRS}" )
endif(OpenCV_FOUND)

add_definitions(-DOPENCV)
add_definitions(-O3 -g)
add_definitions (-DWITH_OPENCV) 


include_directories(
  ${OpenCV_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} 
main.cpp
CDNeuralNet.cpp
CDNeuralNet.hpp
)

target_link_libraries(${PROJECT_NAME}
  pthread
  stdc++
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  mynteye_depth
  ${OpenCV_LIBS}
)