include_directories(${GTEST_INCLUDE_DIRS})

add_executable(TestLayerInterface TestLayerInterface.cpp)
target_link_libraries(TestLayerInterface ${GTEST_BOTH_LIBRARIES})
MESSAGE ("Source path: ${PROJECT_SOURCE_DIR}")

target_include_directories(TestLayerInterface PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_test(NAME TestLayerInterface
         COMMAND TestLayerInterface)

add_executable(TestLinearLayer TestLinearLayer.cpp)
target_link_libraries(TestLinearLayer ${GTEST_BOTH_LIBRARIES})
MESSAGE ("Source path: ${PROJECT_SOURCE_DIR}")

target_include_directories(TestLinearLayer PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_test(NAME TestLinearLayer
        COMMAND TestLinearLayer)

         

