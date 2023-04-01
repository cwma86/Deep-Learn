// Your First C++ Program

#include <iostream>
#include "LayerInterface.h"
#include "eigen3/Eigen/Dense"

int main() {
    LayerInterface testInterface;
    int rows = 3;
    int cols = 3; 
    Eigen::MatrixXd testMatrix(rows,cols); //3 by 3 double precision matrix initialization

    // Let's make it a symmetric matrix
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
            testMatrix(i,j) = (i+1)*(1+j);
    }
    testInterface.setPrevOut(testMatrix);
    std::cout << "testInterface \n" << testInterface.getPrevOut() << std::endl;
    cols = 4;
    Eigen::MatrixXd testMatrix2(rows,cols); //3 by 3 double precision matrix initialization

    // Let's make it a symmetric matrix
    for(int i=0; i<rows; i++)
    {
        for(int j=0; j<cols; j++)
            testMatrix2(i,j) = (i+1)*(1+j);
    }
    testInterface.setPrevOut(testMatrix2);
    std::cout << "testInterface2 \n" << testInterface.getPrevOut() << std::endl;

    return 0;
}