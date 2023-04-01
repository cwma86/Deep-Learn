#ifndef DEEP_LEARN_LAYERINTERFACE_H
#define DEEP_LEARN_LAYERINTERFACE_H
#include <iostream>
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

};
#endif //DEEP_LEARN_LAYERINTERFACE_H