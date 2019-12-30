#ifndef PBOX_H
#define PBOX_H

#include <stdlib.h>
#include <iostream>
#include <opencv2/core/cvstd.hpp>
#include <vector>

using namespace std;
//#define mydataFmt double
#define Num 128
typedef double mydataFmt;


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

class pBox1 {
public:
    vector<vector<vector<mydataFmt>>> pdata;
};

class pRelu1 {
public:
    vector<mydataFmt> pdata;
};

class Weight1 {
public:
    vector<vector<vector<vector<mydataFmt>>>> pdata;
    vector<mydataFmt> pbias;
    int stride;
    int padw;
    int padh;
};

void freepBox(struct pBox *pbox);

void freeWeight(struct Weight *weight);

void freepRelu(struct pRelu *prelu);

#endif