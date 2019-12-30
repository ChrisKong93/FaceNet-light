//
// Created by Lenovo on 2019/10/17.
//

#ifndef MAIN_FACENET_H
#define MAIN_FACENET_H

#include "network.h"


class facenet {
public:
    facenet();

    ~facenet();

    void run(Mat &image, vector<mydataFmt> &o, int count = 1);

private:
    void Stem(Mat &image, pBox *output);

    void Inception_resnet_A(pBox *input, pBox *output, string filepath = "", float scale = 1.0);

    void Reduction_A(pBox *input, pBox *output);

    void Inception_resnet_B(pBox *input, pBox *output, string filepath = "", float scale = 1.0);

    void Reduction_B(pBox *input, pBox *output);

    void Inception_resnet_C(pBox *input, pBox *output, string filepath = "", float scale = 1.0);

    void Inception_resnet_C_None(pBox *input, pBox *output, string filepath = "");

    void AveragePooling(pBox *input, pBox *output);

    void fully_connect(pBox *input, pBox *output, string filepath = "");

    void Flatten(pBox *input, pBox *output);

    void printData(pBox *output);
};

#endif //MAIN_FACENET_H
