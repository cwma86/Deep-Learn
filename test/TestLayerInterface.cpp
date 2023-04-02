#include <gtest/gtest.h>
#include <LayerInterface.h>
#include <eigen3/Eigen/Dense>

//! LayerInterface test class
/*!
  A class made for testing the LayerInterface, by providing dummy implementation
  for the pure virutual class metods
*/
class TestLayer: public LayerInterface{
    virtual Eigen::MatrixXd  forward(Eigen::MatrixXd dataIn) override {
        // No Implementation needed for test
        return dataIn;
    } ;
    virtual std::vector<Eigen::MatrixXd> gradient() override {
        // No Implementation needed for test.
        std::vector<Eigen::MatrixXd> fakeOutput;
        return fakeOutput;
    } ;
};

// Demonstrate some basic assertions.
TEST(LayerInterfaceTest, BasicAssertions)
{
    TestLayer testInterface;
    int rows = 3;
    int cols = 3;
    Eigen::MatrixXd testMatrix(rows, cols); // 3 by 3 double precision matrix initialization

    // Let's make it a symmetric matrix
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            testMatrix(i, j) = (i + 1) * (1 + j);
    }
    testInterface.setPrevOut(testMatrix);
    Eigen::MatrixXd LayerOutput = testInterface.getPrevOut();

    EXPECT_EQ(LayerOutput.rows(), 3);
    EXPECT_EQ(LayerOutput.cols(), 3);

    cols = 4;
    Eigen::MatrixXd testMatrix2(rows, cols); // 3 by 3 double precision matrix initialization

    // Let's make it a symmetric matrix
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
            testMatrix2(i, j) = (i + 1) * (1 + j);
    }
    testInterface.setPrevOut(testMatrix2);
    LayerOutput = testInterface.getPrevOut();
    std::cout << "LayerOutput: \n"
              << LayerOutput << std::endl;

    EXPECT_EQ(LayerOutput.rows(), 3);
    EXPECT_EQ(LayerOutput.cols(), 4);
    EXPECT_EQ(LayerOutput(0, 0), 1);
    EXPECT_EQ(LayerOutput(2, 3), 12);
}
