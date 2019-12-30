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
    for (int rowI = 0; rowI < image.rows; rowI++) {
        for (int colK = 0; colK < image.cols; colK++) {
            if (num == 0) {
                *p = (image.at<Vec3b>(rowI, colK)[0] - 127.5) * 0.0078125;
                *(p + image.rows * image.cols) = (image.at<Vec3b>(rowI, colK)[1] - 127.5) * 0.0078125;
                *(p + 2 * image.rows * image.cols) = (image.at<Vec3b>(rowI, colK)[2] - 127.5) * 0.0078125;
                p++;
            } else {
                double mean, stddev, sqr, stddev_adj;
                int size;
                Mat temp_m, temp_sd;
                meanStdDev(image, temp_m, temp_sd);
                mean = temp_m.at<double>(0, 0);
                stddev = temp_sd.at<double>(0, 0);
                size = image.cols * image.rows * image.channels();
                sqr = sqrt(double(size));

                if (stddev >= 1.0 / sqr) {
                    stddev_adj = stddev;
                } else {
                    stddev_adj = 1.0 / sqr;
                }
//                cout << mean << "|" << stddev << "|" << size << "|" << stddev_adj << "|" << endl;
                for (int i = 0; i < image.rows; i++) {
                    for (int j = 0; j < image.cols; j++) {
                        image.at<uchar>(i, j);
                        *p = (image.at<Vec3b>(i, j)[0] - mean) / stddev_adj;
                        *(p + image.rows * image.cols) = (image.at<Vec3b>(i, j)[1] - mean) / stddev_adj;
                        *(p + 2 * image.rows * image.cols) = (image.at<Vec3b>(i, j)[2] - mean) / stddev_adj;
//                        cout << (image.at<Vec3b>(i, j)[0] - mean) / stddev_adj << endl;
//                        return;
                    }
                }
            }
        }
    }
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
//    if (weight->pad != 0) {
//        pBox *padpbox = new pBox;
//        featurePadInit(outpBox, padpbox, weight->pad, weight->padw, weight->padh);
//        featurePad(outpBox, padpbox, weight->pad, weight->padw, weight->padh);
//        *outpBox = *padpbox;
//    }
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
//    cout << "output->pdata:" << (outpBox->pdata[10]) << endl;
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


void prelu(struct pBox *pbox, mydataFmt *pbias, mydataFmt *prelu_gmma) {
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
    mydataFmt *pg = prelu_gmma;

    long dis = pbox->width * pbox->height;
    for (int channel = 0; channel < pbox->channel; channel++) {
        for (int col = 0; col < dis; col++) {
            *op = *op + *pb;
            *op = (*op > 0) ? (*op) : ((*op) * (*pg));
            op++;
        }
        pb++;
        pg++;
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
                  pbox->width * pbox->height * pbox->channel,
                  weight->lastChannel, weight->selfChannel,
                  outpBox->pdata);
}

void vectorXmatrix(mydataFmt *matrix, mydataFmt *v, int size, int v_w, int v_h, mydataFmt *p) {
    for (int i = 0; i < v_h; i++) {
        p[i] = 0;
        for (int j = 0; j < v_w; j++) {
            p[i] += matrix[j] * v[i * v_w + j];
//            cout << p[i] << endl;
        }
//        cout << p[i] << endl;
//        p[i] = -0.0735729;
//        cout << "...." << endl;
//        break;
    }
//    cout << "...." << endl;
}

void readData(string filename, long dataNumber[], mydataFmt *pTeam[], int length) {
    ifstream in(filename.data());
    string line;
    if (in) {
        int i = 0;
        int count = 0;
        int pos = 0;
        while (getline(in, line)) {
            try {
                if (i < dataNumber[count]) {
                    line.erase(0, 1);
                    pos = line.find(']');
                    line.erase(pos, 1);
                    pos = line.find('\r');
                    if (pos != -1) {
                        line.erase(pos, 1);
                    }
                    *(pTeam[count])++ = atof(line.data());
                } else {
                    count++;
                    if ((length != 0) && (count == length))
                        break;
                    dataNumber[count] += dataNumber[count - 1];
                    line.erase(0, 1);
                    pos = line.find(']');
                    line.erase(pos, 1);
                    pos = line.find('\r');
                    if (pos != -1) {
                        line.erase(pos, 1);
                    }
                    *(pTeam[count])++ = atof(line.data());
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
long initConvAndFc(struct Weight *weight, int schannel, int lchannel, int kersize,
                   int stride, int pad, int w, int h, int padw, int padh) {
    weight->selfChannel = schannel;
    weight->lastChannel = lchannel;
    weight->kernelSize = kersize;
//    if (kersize == 0) {
    weight->h = h;
    weight->w = w;
//    }
//    if (pad == -1) {
    weight->padh = padh;
    weight->padw = padw;
//    }
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

void initpRelu(struct pRelu *prelu, int width) {
    prelu->width = width;
    prelu->pdata = (mydataFmt *) malloc(width * sizeof(mydataFmt));
    if (prelu->pdata == NULL)cout << "prelu apply for memory failed!!!!";
    memset(prelu->pdata, 0, width * sizeof(mydataFmt));
}