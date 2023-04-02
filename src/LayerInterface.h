#ifndef DEEP_LEARN_LAYERINTERFACE_H
#define DEEP_LEARN_LAYERINTERFACE_H
#include <iostream>
#include <vector>
#include <eigen3/Eigen/Dense>


class LayerInterface{
    private:
    Eigen::MatrixXd prevIn_;
    Eigen::MatrixXd prevOut_;
    
    public:
    void setPrevOut(Eigen::MatrixXd prevOut){
        prevOut_ = prevOut;
    }

    Eigen::MatrixXd getPrevOut(){
        return prevOut_;
    }

    void setPrevIn(Eigen::MatrixXd prevIn){
        prevIn_ = prevIn;
    }

    Eigen::MatrixXd getPrevIn(){
        return prevIn_;
    }

    Eigen::MatrixXd backward(Eigen::MatrixXd gradIn){
        // TODO
        std::vector<Eigen::MatrixXd> gradient = this->gradient();
        Eigen::MatrixXd outputMatrix(gradIn.rows(), gradient.front().cols());
        for (int i =0; i < gradIn.rows(); i++){
            // TODO some dot produce
            outputMatrix.row(i) = gradIn.row(i).transpose() * gradient.at(i));
        }
        return gradIn;
    }

    virtual Eigen::MatrixXd forward(Eigen::MatrixXd dataIn) = 0;
    virtual std::vector<Eigen::MatrixXd> gradient() = 0;
};
#endif //DEEP_LEARN_LAYERINTERFACE_H