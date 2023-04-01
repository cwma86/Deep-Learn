enable_testing()
set(GTest_ROOT /usr/src/googletest/googletest)
add_subdirectory(${GTest_ROOT}
        "${CMAKE_CURRENT_BINARY_DIR}/googletest" EXCLUDE_FROM_ALL)
add_executable(hello_test hello_test.cpp)
target_link_libraries(hello_test GTest::gtest GTest::gtest_main)

add_executable(TestLayerInterface TestLayerInterface.cpp)
target_link_libraries(TestLayerInterface GTest::gtest GTest::gtest_main)
MESSAGE ("Source path: ${PROJECT_SOURCE_DIR}")

target_include_directories(TestLayerInterface PRIVATE ${PROJECT_SOURCE_DIR}/src)

add_test(AllTestsInProj hello_test TestLayerInterface)


