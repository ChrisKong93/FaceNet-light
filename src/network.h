#ifndef NETWORK_H
#define NETWORK_H

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <cstring>
#include <string>
#include <math.h>
#include "pBox.h"

using namespace cv;

void addbias(struct pBox *pbox, mydataFmt *pbias);

void image2Matrix(const Mat &image, const struct pBox *pbox, int num = 0);

void maxPooling(const pBox *pbox, pBox *Matrix, int kernelSize, int stride);

void avePooling(const pBox *pbox, pBox *Matrix, int kernelSize, int stride);

void featurePad(const pBox *pbox, pBox *outpBox, const int pad, const int padw = 0, const int padh = 0);

void prelu(struct pBox *pbox, mydataFmt *pbias, mydataFmt *prelu_gmma);

void relu(struct pBox *pbox, mydataFmt *pbias);

void fullconnect(const Weight *weight, const pBox *pbox, pBox *outpBox);

void readData(string filename, long dataNumber[], mydataFmt *pTeam[], int length = 0);

long ConvAndFcInit(struct Weight *weight, int schannel, int lchannel, int kersize, int stride, int pad,
                   int w = 0, int h = 0, int padw = 0, int padh = 0);

void pReluInit(struct pRelu *prelu, int width);

void softmax(const struct pBox *pbox);

void image2MatrixInit(Mat &image, struct pBox *pbox);

void featurePadInit(const pBox *pbox, pBox *outpBox, const int pad, const int padw = 0, const int padh = 0);

void maxPoolingInit(const pBox *pbox, pBox *Matrix, int kernelSize, int stride, int flag = 0);

void avePoolingInit(const pBox *pbox, pBox *Matrix, int kernelSize, int stride);

void convolutionInit(const Weight *weight, pBox *pbox, pBox *outpBox);

void fullconnectInit(const Weight *weight, pBox *outpBox);

bool cmpScore(struct orderScore lsh, struct orderScore rsh);

void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const mydataFmt overlap_threshold,
         string modelname = "Union");

void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width);

void vectorXmatrix(mydataFmt *matrix, mydataFmt *v, int v_w, int v_h, mydataFmt *p);

void convolution(const Weight *weight, const pBox *pbox, pBox *outpBox);

void MeanAndDev(const Mat &image, mydataFmt &p, mydataFmt &q);

void conv_merge(pBox *output, pBox *c1 = 0, pBox *c2 = 0, pBox *c3 = 0, pBox *c4 = 0);

void conv_mergeInit(pBox *output, pBox *c1 = 0, pBox *c2 = 0, pBox *c3 = 0, pBox *c4 = 0);

void mulandaddInit(const pBox *inpbox, const pBox *temppbox, pBox *outpBox);

void mulandadd(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale = 1);

void BatchNormInit(struct BN *beta, struct BN *mean, struct BN *var, int width);

void BatchNorm(struct pBox *pbox, struct BN *beta, struct BN *mean, struct BN *var);

#endif