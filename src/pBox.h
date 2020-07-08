#ifndef PBOX_H
#define PBOX_H

#include <stdlib.h>
#include <iostream>
#include <opencv2/core/cvstd.hpp>
#include <vector>

/**
 * 声明结构体
 */

using namespace std;
//#define mydataFmt double
#define Num 512
//#define Num 128
typedef float mydataFmt;

struct pBox : public cv::String {
    mydataFmt *pdata;
    int width;
    int height;
    int channel;
};

struct BN {
    mydataFmt *pdata;
    int width;
};

struct Weight {
    mydataFmt *pdata;
    mydataFmt *pbias;
    int lastChannel;
    int selfChannel;
    int kernelSize;
    int stride;
    int pad;
    int w;
    int h;
    int padw;
    int padh;
};

void freepBox(struct pBox *pbox);

void freeWeight(struct Weight *weight);

void freeBN(struct BN *bn);

#endif