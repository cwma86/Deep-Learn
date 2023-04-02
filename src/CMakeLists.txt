
set(HEADER_FILES 
    LayerInterface.h 
    LinearLayer.h
    FullyConnectedLayer.h)
find_package(Eigen3 REQUIRED)
link_directories(${Eigen_INCLUDE_DIRS})

add_executable(MyExample main.cpp ${HEADER_FILES})
target_link_libraries(MyExample ${Eigen_LIBRARIES} PRIVATE)
