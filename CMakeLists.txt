PROJECT(SwarmImageSegmenter)
CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

FIND_PACKAGE(CUDA REQUIRED)
FIND_PACKAGE(OpenCV REQUIRED)

LINK_LIBRARIES("-L/usr/local/cuda/lib64 -lcudart" ${OpenCV_LIBS})

INCLUDE(FindCUDA)

CUDA_ADD_EXECUTABLE(sis src/main.cpp src/pso_cluster.cpp src/kernel.cu)
