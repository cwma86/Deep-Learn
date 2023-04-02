#ifndef DEEP_LEARN_FULLYCONNECTEDLAYER_H
#define DEEP_LEARN_FULLYCONNECTEDLAYER_H
#include <iostream>
#include <vector>
#include <memory>
#include <eigen3/Eigen/Dense>
#include "LayerInterface.h"

class FullyConnectedLayer : public LayerInterface
{
private:
    Eigen::MatrixXd weights_;
    Eigen::MatrixXd bias_;
    double eta_;
    void updateWeights(Eigen::MatrixXd gradIn, int epoch = 1)
    {
        Eigen::MatrixXd dJdw = (this->getPrevIn().transpose() * gradIn) / gradIn.rows();
        this->weights_ = this->weights_ - this->eta_ * dJdw;
        // TODO dJdb for bias derivative
    }

public:
    FullyConnectedLayer(
        int sizeIn,
        int sizeOut,
        double weights = 1e-3,
        double bias = 1e-3,
        std::string weightUpdateFunction = "updateWeights",
        double eta = 0.001) : eta_(eta)
    {
        weights_ = Eigen::MatrixXd::Random(sizeIn, sizeOut) * weights;
        std::cout << "weights \n"
                  << weights_ << std::endl;

        bias_ = Eigen::MatrixXd::Random(1, sizeOut) * bias;
        std::cout << "bias \n"
                  << bias_ << std::endl;

        // TODO create a strategy factory for the weight update function options
    }

    Eigen::MatrixXd getWeights() const
    {
        return weights_;
    }

    void setWeights(Eigen::MatrixXd weights)
    {
        weights_ = weights;
    }

    Eigen::MatrixXd getBias() const
    {
        return bias_;
    }

    void setBias(Eigen::MatrixXd bias)
    {
        bias_ = bias;
    }

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd dataIn) override
    {
        this->setPrevIn(dataIn);
        Eigen::MatrixXd dataOut = dataIn * this->weights_ + this->bias_;
        this->setPrevOut(dataOut);
        return dataOut;
    };
    virtual std::vector<Eigen::MatrixXd> gradient() override
    {
        std::vector<Eigen::MatrixXd> grad;
        for (int i = 0; i < this->getPrevIn().rows(); i++){
            Eigen::MatrixXd tmp_grad( this->getWeights().cols(), this->getPrevIn().rows());
            tmp_grad = this->getWeights().transpose();
            grad.push_back(tmp_grad);
        }
        return grad;
    };
};
#endif // DEEP_LEARN_FULLYCONNECTEDLAYER_H