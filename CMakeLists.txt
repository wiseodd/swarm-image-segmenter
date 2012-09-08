project(SwarmImageSegmenter)

find_package(OpenCV REQUIRED)
add_executable(SwarmImageSegmenter "pso_cluster.cpp" "main.cpp")
target_link_libraries(SwarmImageSegmenter ${OpenCV_LIBS})
