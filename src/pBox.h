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
#define Num 128
typedef float mydataFmt;

struct pBox : public cv::String {
    mydataFmt *pdata;
    int width;
    int height;
    int channel;
};

struct pRelu {
    mydataFmt *pdata;
    int width;
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

struct Bbox {
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool exist;
    mydataFmt ppoint[10];
    mydataFmt regreCoord[4];
};

struct orderScore {
    mydataFmt score;
    int oriOrder;
};

void freepBox(struct pBox *pbox);

void freeWeight(struct Weight *weight);

void freepRelu(struct pRelu *prelu);

void freeBN(struct BN *bn);

#endif