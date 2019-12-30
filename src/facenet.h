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

    void run(Mat &image, mydataFmt *o, int count = 1);

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

    void conv_merge(pBox *output, pBox *c1 = 0, pBox *c2 = 0, pBox *c3 = 0, pBox *c4 = 0);

    void conv_mergeInit(pBox *output, pBox *c1 = 0, pBox *c2 = 0, pBox *c3 = 0, pBox *c4 = 0);

    void mulandaddInit(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale);

    void mulandadd(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale = 1);

    void Flatten(pBox *input, pBox *output);

    void printData(pBox *output);



};

#endif //MAIN_FACENET_H
