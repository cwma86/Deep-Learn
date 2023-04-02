#ifndef DEEP_LEARN_LINEARLAYER_H
#define DEEP_LEARN_LINEARLAYER_H
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>
#include "LayerInterface.h"

class LinearLayer : public LayerInterface
{
public:
    virtual Eigen::MatrixXd forward(Eigen::MatrixXd dataIn) override
    {
        this->setPrevIn(dataIn);
        this->setPrevOut(dataIn);
        return dataIn;
    };
    virtual std::vector<Eigen::MatrixXd> gradient() override
    {
         std::vector<Eigen::MatrixXd> grad;
        for (int i = 0; i < this->getPrevOut().rows(); i++)
        {
            Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(this->getPrevOut().cols(), this->getPrevOut().cols());
            grad.push_back(identity);
        }
        return grad;
    };
};
#endif // DEEP_LEARN_LINEARLAYER_H