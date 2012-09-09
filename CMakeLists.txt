project(SwarmImageSegmenter)

find_package(OpenCV REQUIRED)

file(GLOB SRCS "src/*.cpp")
add_executable(SwarmImageSegmenter ${SRCS})

target_link_libraries(SwarmImageSegmenter ${OpenCV_LIBS})
