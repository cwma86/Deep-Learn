MESSAGE ("Source path: ${PROJECT_SOURCE_DIR}")
include_directories(${GTEST_INCLUDE_DIRS})

add_executable(TestLayerInterface TestLayerInterface.cpp)
target_link_libraries(TestLayerInterface ${GTEST_BOTH_LIBRARIES})

target_include_directories(TestLayerInterface PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_test(NAME TestLayerInterface
         COMMAND TestLayerInterface)

add_executable(TestLinearLayer TestLinearLayer.cpp)
target_link_libraries(TestLinearLayer ${GTEST_BOTH_LIBRARIES})

target_include_directories(TestLinearLayer PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_test(NAME TestLinearLayer
        COMMAND TestLinearLayer)

add_executable(TestFullyConnectedLayer TestFullyConnectedLayer.cpp)
target_link_libraries(TestFullyConnectedLayer ${GTEST_BOTH_LIBRARIES})

target_include_directories(TestFullyConnectedLayer PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_test(NAME TestFullyConnectedLayer
        COMMAND TestFullyConnectedLayer)
        
                 

