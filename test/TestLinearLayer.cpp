#include <gtest/gtest.h>
#include <LinearLayer.h>
#include <eigen3/Eigen/Dense>

// Demonstrate some basic assertions.
TEST(LinearLayerTest, ForwardTest)
{
    LinearLayer testInterface;
    int rows = 1;
    int cols = 6;
    Eigen::MatrixXd testMatrix(rows, cols); // 3 by 3 double precision matrix initialization

    testMatrix << 19.0, 0.0, 27.9, 0.0, 1.0, 4.0;
    Eigen::MatrixXd forward_out = testInterface.forward(testMatrix);
    std::cout << "forward_out \n" << forward_out << std::endl;
    EXPECT_EQ(forward_out.rows(), rows);
    EXPECT_EQ(forward_out.cols(), cols);
    EXPECT_EQ(forward_out, testMatrix);
}

TEST(LinearLayerTest, GradientTest)
{
    LinearLayer testInterface;
    int rows = 1;
    int cols = 6;
    Eigen::MatrixXd testMatrix(rows, cols); // build a 1x6 matrix (array)

    testMatrix << 19.0, 0.0, 27.9, 0.0, 1.0, 4.0;
    Eigen::MatrixXd forward_out = testInterface.forward(testMatrix);
    std::cout << "forward_out \n" << forward_out << std::endl;
    EXPECT_EQ(forward_out.rows(), rows);
    EXPECT_EQ(forward_out.cols(), cols);
    EXPECT_EQ(forward_out, testMatrix);
    std::vector<Eigen::MatrixXd> gradient = testInterface.gradient();
    for (int i = 0; i < gradient.size(); i++ ){
        std::cout << "gradient["<< i <<"]\n" << gradient.at(i) << std::endl;
        EXPECT_EQ(
            Eigen::MatrixXd::Identity(forward_out.cols(), forward_out.cols()),
            gradient.at(i)
        );

    }

}

TEST(LinearLayerTest, GradientTest2)
{
    LinearLayer testInterface;
    int rows = 2;
    int cols = 4;
    Eigen::MatrixXd testMatrix(rows, cols); 

    testMatrix << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
    Eigen::MatrixXd forward_out = testInterface.forward(testMatrix);
    std::cout << "forward_out \n" << forward_out << std::endl;
    EXPECT_EQ(forward_out.rows(), rows);
    EXPECT_EQ(forward_out.cols(), cols);
    EXPECT_EQ(forward_out, testMatrix);
    std::vector<Eigen::MatrixXd> gradient = testInterface.gradient();
    for (int i = 0; i < gradient.size(); i++ ){
        std::cout << "gradient["<< i <<"]\n" << gradient.at(i) << std::endl;
        EXPECT_EQ(
            Eigen::MatrixXd::Identity(forward_out.cols(), forward_out.cols()),
            gradient.at(i)
        );

    }

}
