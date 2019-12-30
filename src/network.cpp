#include "network.h"

void addbias(struct pBox *pbox, mydataFmt *pbias) {
    if (pbox->pdata == NULL) {
        cout << "Relu feature is NULL!!" << endl;
        return;
    }
    if (pbias == NULL) {
        cout << "the  Relu bias is NULL!!" << endl;
        return;
    }
    mydataFmt *op = pbox->pdata;
    mydataFmt *pb = pbias;

    long dis = pbox->width * pbox->height;
    for (int channel = 0; channel < pbox->channel; channel++) {
        for (int col = 0; col < dis; col++) {
            *op = *op + *pb;
            op++;
        }
        pb++;
    }
}

void image2MatrixInit(Mat &image, struct pBox *pbox) {
    if ((image.data == NULL) || (image.type() != CV_8UC3)) {
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }
    pbox->channel = image.channels();
    pbox->height = image.rows;
    pbox->width = image.cols;

    pbox->pdata = (mydataFmt *) malloc(pbox->channel * pbox->height * pbox->width * sizeof(mydataFmt));
    if (pbox->pdata == NULL)cout << "the image2MatrixInit failed!!" << endl;
    memset(pbox->pdata, 0, pbox->channel * pbox->height * pbox->width * sizeof(mydataFmt));
}

void image2Matrix(const Mat &image, const struct pBox *pbox, int num) {
    if ((image.data == NULL) || (image.type() != CV_8UC3)) {
        cout << "image's type is wrong!!Please set CV_8UC3" << endl;
        return;
    }
    if (pbox->pdata == NULL) {
        return;
    }
    mydataFmt *p = pbox->pdata;
    double sqr, stddev_adj;
    int size;
    mydataFmt mymean, mystddev;
    // prewhiten
    if (num != 0) {
        MeanAndDev(image, mymean, mystddev);
        cout << mymean << "----" << mystddev << endl;
        size = image.cols * image.rows * image.channels();
        sqr = sqrt(double(size));
        if (mystddev >= 1.0 / sqr) {
            stddev_adj = mystddev;
        } else {
            stddev_adj = 1.0 / sqr;
        }
    }
    for (int rowI = 0; rowI < image.rows; rowI++) {
        for (int colK = 0; colK < image.cols; colK++) {
            if (num == 0) {
                *p = (image.at<Vec3b>(rowI, colK)[2] - 127.5) * 0.0078125;
                *(p + image.rows * image.cols) = (image.at<Vec3b>(rowI, colK)[1] - 127.5) * 0.0078125;
                *(p + 2 * image.rows * image.cols) = (image.at<Vec3b>(rowI, colK)[0] - 127.5) * 0.0078125;
                p++;
            } else {
                // brg2rgb
                *(p + 0 * image.rows * image.cols) = (image.at<Vec3b>(rowI, colK)[2] - mymean) / stddev_adj;
                *(p + 1 * image.rows * image.cols) = (image.at<Vec3b>(rowI, colK)[1] - mymean) / stddev_adj;
                *(p + 2 * image.rows * image.cols) = (image.at<Vec3b>(rowI, colK)[0] - mymean) / stddev_adj;
                p++;
            }
        }
    }
}

void MeanAndDev(const Mat &image, mydataFmt &p, mydataFmt &q) {
    mydataFmt meansum = 0, stdsum = 0;
    for (int rowI = 0; rowI < image.rows; rowI++) {
        for (int colK = 0; colK < image.cols; colK++) {
            meansum += image.at<Vec3b>(rowI, colK)[0] + image.at<Vec3b>(rowI, colK)[1] + image.at<Vec3b>(rowI, colK)[2];
        }
    }
    p = meansum / (image.cols * image.rows * image.channels());
    for (int rowI = 0; rowI < image.rows; rowI++) {
        for (int colK = 0; colK < image.cols; colK++) {
            stdsum += pow((image.at<Vec3b>(rowI, colK)[0] - p), 2) +
                      pow((image.at<Vec3b>(rowI, colK)[1] - p), 2) +
                      pow((image.at<Vec3b>(rowI, colK)[2] - p), 2);
        }
    }
    q = sqrt(stdsum / (image.cols * image.rows * image.channels()));
}

void featurePadInit(const pBox *pbox, pBox *outpBox, const int pad, const int padw, const int padh) {
    if (pad < -1) {
        cout << "the data needn't to pad,please check you network!" << endl;
        return;
    }
    outpBox->channel = pbox->channel;
    if (pad == -1) {
        outpBox->height = pbox->height + 2 * padh;
        outpBox->width = pbox->width + 2 * padw;
    } else {
        outpBox->height = pbox->height + 2 * pad;
        outpBox->width = pbox->width + 2 * pad;
    }
    long RowByteNum = outpBox->width * sizeof(mydataFmt);
    outpBox->pdata = (mydataFmt *) malloc(outpBox->channel * outpBox->height * RowByteNum);
    if (outpBox->pdata == NULL)cout << "the featurePadInit is failed!!" << endl;
    memset(outpBox->pdata, 0, outpBox->channel * outpBox->height * RowByteNum);
}

void featurePad(const pBox *pbox, pBox *outpBox, const int pad, const int padw, const int padh) {
    mydataFmt *p = outpBox->pdata;
    mydataFmt *pIn = pbox->pdata;
    if (pad == -1) {
        for (int row = 0; row < outpBox->channel * outpBox->height; row++) {
            if ((row % outpBox->height) < padh || (row % outpBox->height > (outpBox->height - padh - 1))) {
                p += outpBox->width;
                continue;
            }
            p += padw;
            memcpy(p, pIn, pbox->width * sizeof(mydataFmt));
            p += pbox->width + padw;
            pIn += pbox->width;
        }
    } else {
        for (int row = 0; row < outpBox->channel * outpBox->height; row++) {
            if ((row % outpBox->height) < pad || (row % outpBox->height > (outpBox->height - pad - 1))) {
                p += outpBox->width;
                continue;
            }
            p += pad;
            memcpy(p, pIn, pbox->width * sizeof(mydataFmt));
            p += pbox->width + pad;
            pIn += pbox->width;
        }
    }
}

void convolutionInit(const Weight *weight, pBox *pbox, pBox *outpBox) {
    outpBox->channel = weight->selfChannel;
//    ((imginputh - ckh + 2 * ckpad) / stride) + 1;
    if (weight->kernelSize == 0) {
        outpBox->width = ((pbox->width - weight->w + 2 * weight->padw) / weight->stride) + 1;
//        outpBox->width = (pbox->width - weight->w) / weight->stride + 1;
//        outpBox->height = (pbox->height - weight->h) / weight->stride + 1;
        outpBox->height = (pbox->height - weight->h + 2 * weight->padh) / weight->stride + 1;
    } else {
        outpBox->width = ((pbox->width - weight->kernelSize + 2 * weight->pad) / weight->stride) + 1;
        outpBox->height = ((pbox->height - weight->kernelSize + 2 * weight->pad) / weight->stride) + 1;
    }
//    cout << outpBox->pdata << endl;
    outpBox->pdata = (mydataFmt *) malloc(outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
//    cout << outpBox->pdata << endl;
    if (outpBox->pdata == NULL)cout << "the convolutionInit is failed!!" << endl;
    memset(outpBox->pdata, 0, outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
    if (weight->pad != 0) {
        pBox *padpbox = new pBox;
        featurePadInit(pbox, padpbox, weight->pad, weight->padw, weight->padh);
        featurePad(pbox, padpbox, weight->pad, weight->padw, weight->padh);
        *pbox = *padpbox;
    }
}

void convolution(const Weight *weight, const pBox *pbox, pBox *outpBox) {
    int ckh, ckw, ckd, stride, cknum, ckpad, imginputh, imginputw, imginputd, Nh, Nw;
    mydataFmt *ck, *imginput;
//    float *output = outpBox->pdata;
    float temp;
    ck = weight->pdata;
    if (weight->kernelSize == 0) {
        ckh = weight->h;
        ckw = weight->w;
    } else {
        ckh = weight->kernelSize;
        ckw = weight->kernelSize;
    }
    ckd = weight->lastChannel;
    cknum = weight->selfChannel;
    ckpad = weight->pad;
    stride = weight->stride;
    imginput = pbox->pdata;
    imginputh = pbox->height;
    imginputw = pbox->width;
    imginputd = pbox->channel;
    Nh = outpBox->height;
    Nw = outpBox->width;
//    Nh = ((imginputh - ckh + 2 * ckpad) / stride) + 1;
//    Nw = ((imginputw - ckw + 2 * ckpad) / stride) + 1;
    for (int i = 0; i < cknum; ++i) {
        for (int j = 0; j < Nh; j++) {
            for (int k = 0; k < Nw; k++) {
                temp = 0;

                for (int m = 0; m < ckd; ++m) {
                    for (int n = 0; n < ckh; ++n) {
                        for (int i1 = 0; i1 < ckw; ++i1) {
                            temp += imginput[(j * stride + n) * imginputw
                                             + (k * stride + i1)
                                             + m * imginputh * imginputw]
                                    * ck[i * ckh * ckw * ckd + m * ckh * ckw + n * ckw + i1];
                        }
                    }
                }
                //按照顺序存储
                outpBox->pdata[i * outpBox->height * outpBox->width + j * outpBox->width + k] = temp;
            }
        }
    }
}

void maxPoolingInit(const pBox *pbox, pBox *Matrix, int kernelSize, int stride, int flag) {
    if (flag == 1) {
        Matrix->width = floor((float) (pbox->width - kernelSize) / stride + 1);
        Matrix->height = floor((float) (pbox->height - kernelSize) / stride + 1);
    } else {
        Matrix->width = ceil((float) (pbox->width - kernelSize) / stride + 1);
        Matrix->height = ceil((float) (pbox->height - kernelSize) / stride + 1);
    }
    Matrix->channel = pbox->channel;
    Matrix->pdata = (mydataFmt *) malloc(Matrix->channel * Matrix->width * Matrix->height * sizeof(mydataFmt));
    if (Matrix->pdata == NULL)cout << "the maxPoolingI nit is failed!!" << endl;
    memset(Matrix->pdata, 0, Matrix->channel * Matrix->width * Matrix->height * sizeof(mydataFmt));
}

void maxPooling(const pBox *pbox, pBox *Matrix, int kernelSize, int stride) {
    if (pbox->pdata == NULL) {
        cout << "the feature2Matrix pbox is NULL!!" << endl;
        return;
    }
    mydataFmt *p = Matrix->pdata;
    mydataFmt *pIn;
    mydataFmt *ptemp;
    mydataFmt maxNum = 0;
    if ((pbox->width - kernelSize) % stride == 0 && (pbox->height - kernelSize) % stride == 0) {
        for (int row = 0; row < Matrix->height; row++) {
            for (int col = 0; col < Matrix->width; col++) {
                pIn = pbox->pdata + row * stride * pbox->width + col * stride;
                for (int channel = 0; channel < pbox->channel; channel++) {
                    ptemp = pIn + channel * pbox->height * pbox->width;
                    maxNum = *ptemp;
                    for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
                        for (int i = 0; i < kernelSize; i++) {
                            if (maxNum < *(ptemp + i + kernelRow * pbox->width))
                                maxNum = *(ptemp + i + kernelRow * pbox->width);
                        }
                    }
                    *(p + channel * Matrix->height * Matrix->width) = maxNum;
                }
                p++;
            }
        }
    } else {
        int diffh = 0, diffw = 0;
        for (int channel = 0; channel < pbox->channel; channel++) {
            pIn = pbox->pdata + channel * pbox->height * pbox->width;
            for (int row = 0; row < Matrix->height; row++) {
                for (int col = 0; col < Matrix->width; col++) {
                    ptemp = pIn + row * stride * pbox->width + col * stride;
                    maxNum = *ptemp;
                    diffh = row * stride - pbox->height + 1;
                    diffw = col * stride - pbox->width + 1;
                    for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
                        if ((kernelRow + diffh) > 0)break;
                        for (int i = 0; i < kernelSize; i++) {
                            if ((i + diffw) > 0)break;
                            if (maxNum < *(ptemp + i + kernelRow * pbox->width))
                                maxNum = *(ptemp + i + kernelRow * pbox->width);
                        }
                    }
                    *p++ = maxNum;
                }
            }
        }
    }
}

void avePoolingInit(const pBox *pbox, pBox *Matrix, int kernelSize, int stride) {
    Matrix->width = ceil((float) (pbox->width - kernelSize) / stride + 1);
    Matrix->height = ceil((float) (pbox->height - kernelSize) / stride + 1);
    Matrix->channel = pbox->channel;
    Matrix->pdata = (mydataFmt *) malloc(Matrix->channel * Matrix->width * Matrix->height * sizeof(mydataFmt));
    if (Matrix->pdata == NULL)cout << "the maxPoolingInit is failed!!" << endl;
    memset(Matrix->pdata, 0, Matrix->channel * Matrix->width * Matrix->height * sizeof(mydataFmt));
}

void avePooling(const pBox *pbox, pBox *Matrix, int kernelSize, int stride) {
    if (pbox->pdata == NULL) {
        cout << "the feature2Matrix pbox is NULL!!" << endl;
        return;
    }
    mydataFmt *p = Matrix->pdata;
    mydataFmt *pIn;
    mydataFmt *ptemp;
    mydataFmt sumNum = 0;
    if ((pbox->width - kernelSize) % stride == 0 && (pbox->height - kernelSize) % stride == 0) {
        for (int row = 0; row < Matrix->height; row++) {
            for (int col = 0; col < Matrix->width; col++) {
                pIn = pbox->pdata + row * stride * pbox->width + col * stride;

                for (int channel = 0; channel < pbox->channel; channel++) {

                    ptemp = pIn + channel * pbox->height * pbox->width;
                    sumNum = 0;
                    for (int kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
                        for (int i = 0; i < kernelSize; i++) {
                            sumNum += *(ptemp + i + kernelRow * pbox->width);
                        }
                    }
                    *(p + channel * Matrix->height * Matrix->width) = sumNum / (kernelSize * kernelSize);
                }
                p++;
            }
        }
    }
}

/**
 * 激活函数 没有系数
 * @param pbox
 * @param pbias
 */
void relu(struct pBox *pbox, mydataFmt *pbias) {
    if (pbox->pdata == NULL) {
        cout << "the  Relu feature is NULL!!" << endl;
        return;
    }
    if (pbias == NULL) {
        cout << "the  Relu bias is NULL!!" << endl;
        return;
    }
    mydataFmt *op = pbox->pdata;
    mydataFmt *pb = pbias;

    long dis = pbox->width * pbox->height;
    for (int channel = 0; channel < pbox->channel; channel++) {
        for (int col = 0; col < dis; col++) {
            *op = *op + *pb;
            *op = (*op > 0) ? (*op) : ((*op) * 0);
            op++;
        }
        pb++;
    }
}

void fullconnectInit(const Weight *weight, pBox *outpBox) {
    outpBox->channel = weight->selfChannel;
    outpBox->width = 1;
    outpBox->height = 1;
    outpBox->pdata = (mydataFmt *) malloc(weight->selfChannel * sizeof(mydataFmt));
    if (outpBox->pdata == NULL)cout << "the fullconnectInit is failed!!" << endl;
    memset(outpBox->pdata, 0, weight->selfChannel * sizeof(mydataFmt));
}

void fullconnect(const Weight *weight, const pBox *pbox, pBox *outpBox) {
    if (pbox->pdata == NULL) {
        cout << "the fc feature is NULL!!" << endl;
        return;
    }
    if (weight->pdata == NULL) {
        cout << "the fc weight is NULL!!" << endl;
        return;
    }
    memset(outpBox->pdata, 0, weight->selfChannel * sizeof(mydataFmt));
    //Y←αAX + βY    β must be 0(zero)
    //               row         no trans         A's row               A'col
    //cblas_sgemv(CblasRowMajor, CblasNoTrans, weight->selfChannel, weight->lastChannel, 1, weight->pdata, weight->lastChannel, pbox->pdata, 1, 0, outpBox->pdata, 1);
    vectorXmatrix(pbox->pdata, weight->pdata,
                  weight->lastChannel, weight->selfChannel,
                  outpBox->pdata);
}

void vectorXmatrix(mydataFmt *matrix, mydataFmt *v, int v_w, int v_h, mydataFmt *p) {
    for (int i = 0; i < v_h; i++) {
        p[i] = 0;
        for (int j = 0; j < v_w; j++) {
            p[i] += matrix[j] * v[i * v_w + j];
        }
    }
}

void readData(string filename, long dataNumber[], mydataFmt *pTeam[], int length) {
    ifstream in(filename.data());
    string line;
    long temp = dataNumber[0];
    if (in) {
        int i = 0;
        int count = 0;
        int pos = 0;
        while (getline(in, line)) {
            try {
                if (i < temp) {
                    line.erase(0, 1);
                    pos = line.find(']');
                    line.erase(pos, 1);
                    pos = line.find('\r');
                    if (pos != -1) {
                        line.erase(pos, 1);
                    }
                    if (dataNumber[count] != 0) {
                        *(pTeam[count])++ = atof(line.data());
                    }
                } else {
                    count++;
                    if ((length != 0) && (count == length))
                        break;
                    temp += dataNumber[count];
                    line.erase(0, 1);
                    pos = line.find(']');
                    line.erase(pos, 1);
                    pos = line.find('\r');
                    if (pos != -1) {
                        line.erase(pos, 1);
                    }
                    if (dataNumber[count] != 0) {
                        *(pTeam[count])++ = atof(line.data());
                    }
                }
                i++;
            }
            catch (exception &e) {
                cout << " error " << i << endl;
                return;
            }
        }
    } else {
        cout << "no such file" << filename << endl;
    }
}

//									w			  sc			 lc			ks				s		p    kw       kh
long ConvAndFcInit(struct Weight *weight, int schannel, int lchannel, int kersize,
                   int stride, int pad, int w, int h, int padw, int padh) {
    weight->selfChannel = schannel;
    weight->lastChannel = lchannel;
    weight->kernelSize = kersize;
    weight->h = h;
    weight->w = w;
    weight->padh = padh;
    weight->padw = padw;
    weight->stride = stride;
    weight->pad = pad;
    weight->pbias = (mydataFmt *) malloc(schannel * sizeof(mydataFmt));
    if (weight->pbias == NULL)cout << "Memory request not successful!!!";
    memset(weight->pbias, 0, schannel * sizeof(mydataFmt));
    long byteLenght;
    if (kersize == 0) {
        byteLenght = weight->selfChannel * weight->lastChannel * weight->h * weight->w;
    } else {
        byteLenght = weight->selfChannel * weight->lastChannel * weight->kernelSize * weight->kernelSize;
    }
    weight->pdata = (mydataFmt *) malloc(byteLenght * sizeof(mydataFmt));
    if (weight->pdata == NULL)cout << "Memory request not successful!!!";
    memset(weight->pdata, 0, byteLenght * sizeof(mydataFmt));
    return byteLenght;
}

void conv_mergeInit(pBox *output, pBox *c1, pBox *c2, pBox *c3, pBox *c4) {
    output->channel = 0;
    output->height = c1->height;
    output->width = c1->width;
    if (c1 != 0) {
        output->channel = c1->channel;
        if (c2 != 0) {
            output->channel += c2->channel;
            if (c3 != 0) {
                output->channel += c3->channel;
                if (c4 != 0) {
                    output->channel += c4->channel;
                }
            }
        }
    }
    output->pdata = (mydataFmt *) malloc(output->width * output->height * output->channel * sizeof(mydataFmt));
    if (output->pdata == NULL)cout << "the conv_mergeInit is failed!!" << endl;
    memset(output->pdata, 0, output->width * output->height * output->channel * sizeof(mydataFmt));
}

void conv_merge(pBox *output, pBox *c1, pBox *c2, pBox *c3, pBox *c4) {
//    cout << "output->channel:" << output->channel << endl;
    if (c1 != 0) {
        long count1 = c1->height * c1->width * c1->channel;
        //output->pdata = c1->pdata;
        for (long i = 0; i < count1; i++) {
            output->pdata[i] = c1->pdata[i];
        }
        if (c2 != 0) {
            long count2 = c2->height * c2->width * c2->channel;
            for (long i = 0; i < count2; i++) {
                output->pdata[count1 + i] = c2->pdata[i];
            }
            if (c3 != 0) {
                long count3 = c3->height * c3->width * c3->channel;
                for (long i = 0; i < count3; i++) {
                    output->pdata[count1 + count2 + i] = c3->pdata[i];
                }
                if (c4 != 0) {
                    long count4 = c4->height * c4->width * c4->channel;
                    for (long i = 0; i < count4; i++) {
                        output->pdata[count1 + count2 + count3 + i] = c4->pdata[i];
                    }
                }
            }
        }
    } else { cout << "conv_mergeInit" << endl; }
}

void mulandaddInit(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale) {
    outpBox->channel = temppbox->channel;
    outpBox->width = temppbox->width;
    outpBox->height = temppbox->height;
    outpBox->pdata = (mydataFmt *) malloc(outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
    if (outpBox->pdata == NULL)cout << "the mulandaddInit is failed!!" << endl;
    memset(outpBox->pdata, 0, outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
}

void mulandadd(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale) {
    mydataFmt *ip = inpbox->pdata;
    mydataFmt *tp = temppbox->pdata;
    mydataFmt *op = outpBox->pdata;
    long dis = inpbox->width * inpbox->height * inpbox->channel;
    for (long i = 0; i < dis; i++) {
        op[i] = ip[i] + tp[i] * scale;
    }
}

void BatchNormInit(struct BN *var, struct BN *mean, struct BN *beta, int width) {
    var->width = width;
    var->pdata = (mydataFmt *) malloc(width * sizeof(mydataFmt));
    if (var->pdata == NULL)cout << "prelu apply for memory failed!!!!";
    memset(var->pdata, 0, width * sizeof(mydataFmt));

    mean->width = width;
    mean->pdata = (mydataFmt *) malloc(width * sizeof(mydataFmt));
    if (mean->pdata == NULL)cout << "prelu apply for memory failed!!!!";
    memset(mean->pdata, 0, width * sizeof(mydataFmt));

    beta->width = width;
    beta->pdata = (mydataFmt *) malloc(width * sizeof(mydataFmt));
    if (beta->pdata == NULL)cout << "prelu apply for memory failed!!!!";
    memset(beta->pdata, 0, width * sizeof(mydataFmt));
}

void BatchNorm(struct pBox *pbox, struct BN *var, struct BN *mean, struct BN *beta) {
    if (pbox->pdata == NULL) {
        cout << "Relu feature is NULL!!" << endl;
        return;
    }
    if ((var->pdata == NULL) || (mean->pdata == NULL) || (beta->pdata == NULL)) {
        cout << "the  BatchNorm bias is NULL!!" << endl;
        return;
    }
    mydataFmt *pp = pbox->pdata;
    mydataFmt *vp = var->pdata;
    mydataFmt *mp = mean->pdata;
    mydataFmt *bp = beta->pdata;
    int gamma = 1;
    float epsilon = 0.001;
    long dis = pbox->width * pbox->height;
    mydataFmt temp = 0;
    for (int channel = 0; channel < pbox->channel; channel++) {
        temp = gamma / sqrt(((vp[channel]) + epsilon));
        for (int col = 0; col < dis; col++) {
            *pp = temp * (*pp) + ((bp[channel]) - temp * (mp[channel]));
            pp++;
        }
    }
}