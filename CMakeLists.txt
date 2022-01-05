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

## Specify libraries to link a library or executable target against
# link_directories(/home/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64

target_link_libraries(${PROJECT_NAME}
  pthread
  stdc++
  ${Boost_LIBRARIES}
  ${OpenCV_LIBRARIES}
  mynteye_depth
  ${OpenCV_LIBS}
)


# add_library(LIBeSPDI SHARED IMPORTED )
# set_target_properties(LIBeSPDI PROPERTIES IMPORTED_LOCATION ${CMAKE_CURRENT_LIST_DIR}/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64/libeSPDI.so)
# target_link_libraries(eSPDI SHARED LIBeSPDI )



# # use, i.e. don't skip the full RPATH for the build tree
# set(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# set(CMAKE_INSTALL_RPATH "~/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64")
# # add the automatically determined parts of the RPATH
# # which point to directories outside the build tree to the install RPATH
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)



# include_directories(
#     ~/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64
# )

# set(SOURCES main.cpp ~/MYNT-EYE-D-SDK/3rdparty/eSPDI/linux/x64/libeSPDI.so)
# add_executable(libesPDI ${SOURCES})