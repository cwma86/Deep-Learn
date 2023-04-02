#include <gtest/gtest.h>
#include <FullyConnectedLayer.h>
#include <eigen3/Eigen/Dense>

// Demonstrate some basic assertions.
TEST(FullyConnectedLayerTest, ForwardTest)
{
    // std::srand((unsigned int) time(0));
    std::srand(1);// seed the random number generator

    FullyConnectedLayer testLayer(
        6,
        2
    );

    // Update the weights
    Eigen::MatrixXd weights(6,2);
    weights << 9.7627e-06, 4.30378e-05, 
               2.0553e-05, 8.97663e-06,
              -1.5269e-05, 2.91788e-05,
              -1.2482e-05, 7.83546e-05,
              9.27325e-05, -2.33116e-05,
              5.83450e-05, 5.778983e-06;
    testLayer.setWeights(weights);

    Eigen::MatrixXd bias(1,2);
    bias << 1.36089e-05, 8.5119e-05;
    testLayer.setBias(bias);

    // Build an input matrix
    int rows = 1;
    int cols = 6;
    Eigen::MatrixXd testMatrix(rows, cols); // 3 by 3 double precision matrix initialization

    testMatrix << 19.0, 0.0, 27.9, 0.0, 1.0, 4.0;
    Eigen::MatrixXd forward_out = testLayer.forward(testMatrix);
    std::cout << "forward_out \n" << forward_out << std::endl;
    Eigen::MatrixXd expectedOut(1,2);
    expectedOut << 9.92076e-05, 0.00171673;
    std::cout << "expectedOut \n" << expectedOut << std::endl;
    EXPECT_TRUE(forward_out.isApprox(expectedOut, 1e-4 ));
    std::vector<Eigen::MatrixXd> gradient = testLayer.gradient();
    for (int i = 0; i < gradient.size(); i++ ){
        std::cout << "gradient["<< i <<"]\n" << gradient.at(i) << std::endl;
        // TODO need to add check on gradient values
    }

}
