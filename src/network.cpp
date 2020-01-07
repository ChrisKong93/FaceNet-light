#include "network.h"

/**
 * 卷积以后偏移
 * @param pbox　feature map
 * @param pbias　偏移量
 */
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

/**
 * mat图片转成pbox结构体初始化
 * @param image　mat格式的图片
 * @param pbox　结构体pbox
 */
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

/**
 * mat图片转成pbox结构体
 * @param image　mat格式的图片
 * @param pbox　结构体pbox
 * @param num　选择mtcnn还是facenet 0-mtcnn  非0-facenet  缺省为０
 */
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

/**
 * 求图片像素的平均值和标准差
 * @param image 图片
 * @param p 平均值
 * @param q 标准差
 */
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

/**
 * 卷积补偿初始化
 * @param pbox　输入feature map
 * @param outpBox　输出feature map
 * @param pad　补偿 正方形算子(-1为不规则补偿，0为不需要补偿)
 * @param padw 补偿　不规则算子的宽度
 * @param padh　补偿　不规则算子的高度
 */
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

/**
 * 卷积补偿
 * @param pbox　输入feature map
 * @param outpBox 输出feature map
 * @param pad　补偿 正方形算子(-1为不规则补偿，0为不需要补偿)
 * @param padw 补偿　不规则算子的宽度
 * @param padh　补偿　不规则算子的高度
 */
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

/**
 * 卷积初始化
 * @param weight 卷积权重
 * @param pbox　输入feature map
 * @param outpBox 输出feature map
 */
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

/**
 * 卷积
 * @param weight 卷积权重
 * @param pbox　输入feature map
 * @param outpBox 输出feature map
 */
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

/**
 * 最大值池化初始化
 * @param pbox 输入feature map
 * @param Matrix 输出feature map
 * @param kernelSize 池化算子大小
 * @param stride　步长
 * @param flag　标志位
 */
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

/**
 * 最大值池化
 * @param pbox 输入feature map
 * @param Matrix 输出feature map
 * @param kernelSize 池化算子大小
 * @param stride　步长
 */
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

/**
 * 平均值池化初始化
 * @param pbox 输入feature map
 * @param Matrix 输出feature map
 * @param kernelSize 池化算子大小
 * @param stride　步长
 */
void avePoolingInit(const pBox *pbox, pBox *Matrix, int kernelSize, int stride) {
    Matrix->width = ceil((float) (pbox->width - kernelSize) / stride + 1);
    Matrix->height = ceil((float) (pbox->height - kernelSize) / stride + 1);
    Matrix->channel = pbox->channel;
    Matrix->pdata = (mydataFmt *) malloc(Matrix->channel * Matrix->width * Matrix->height * sizeof(mydataFmt));
    if (Matrix->pdata == NULL)cout << "the maxPoolingInit is failed!!" << endl;
    memset(Matrix->pdata, 0, Matrix->channel * Matrix->width * Matrix->height * sizeof(mydataFmt));
}

/**
 * 平均值池化
 * @param pbox 输入feature map
 * @param Matrix 输出feature map
 * @param kernelSize 池化算子大小
 * @param stride　步长
 */
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
 * 激活函数 有系数 初始化
 * @param prelu　激活函数权重
 * @param width　长度
 */
void pReluInit(struct pRelu *prelu, int width) {
    prelu->width = width;
    prelu->pdata = (mydataFmt *) malloc(width * sizeof(mydataFmt));
    if (prelu->pdata == NULL)cout << "prelu apply for memory failed!!!!";
    memset(prelu->pdata, 0, width * sizeof(mydataFmt));
}

/**
 * 激活函数 有系数
 * @param pbox　输入feature
 * @param pbias 偏移
 * @param prelu_gmma　激活函数权重
 */
void prelu(struct pBox *pbox, mydataFmt *pbias, mydataFmt *prelu_gmma) {
    if (pbox->pdata == NULL) {
        cout << "the  pRelu feature is NULL!!" << endl;
        return;
    }
    if (pbias == NULL) {
        cout << "the  pRelu bias is NULL!!" << endl;
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

/**
 * 激活函数 没有系数
 * @param pbox　输入feature
 * @param pbias 偏移
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

/**
 * 全连接初始化
 * @param weight　权重参数
 * @param outpBox　输出feature map
 */
void fullconnectInit(const Weight *weight, pBox *outpBox) {
    outpBox->channel = weight->selfChannel;
    outpBox->width = 1;
    outpBox->height = 1;
    outpBox->pdata = (mydataFmt *) malloc(weight->selfChannel * sizeof(mydataFmt));
    if (outpBox->pdata == NULL)cout << "the fullconnectInit is failed!!" << endl;
    memset(outpBox->pdata, 0, weight->selfChannel * sizeof(mydataFmt));
}

/**
 * 全连接
 * @param weight　权重参数
 * @param pbox 　输入feature map
 * @param outpBox　输出feature map
 */
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

/**
 * 一维数组与二位矩阵相乘
 * @param matrix　输入feature map
 * @param v 权重
 * @param v_w　权重矩阵的宽度
 * @param v_h　权重矩阵的高度
 * @param p　输出feature map
 */
void vectorXmatrix(mydataFmt *matrix, mydataFmt *v, int v_w, int v_h, mydataFmt *p) {
    for (int i = 0; i < v_h; i++) {
        p[i] = 0;
        for (int j = 0; j < v_w; j++) {
            p[i] += matrix[j] * v[i * v_w + j];
        }
    }
}

/**
 * 读取模型文件
 * @param filename　文件路径
 * @param dataNumber　参数个数数组
 * @param pTeam　变量数组
 * @param length
 */
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

/**
 * 卷积和全连接初始化
 * @param weight　权重
 * @param schannel　卷积核个数
 * @param lchannel　上一层feature map个数
 * @param kersize　卷积核大小
 * @param stride　卷积步长
 * @param pad　卷积是否补偿
 * @param w　卷积核宽度
 * @param h　卷积核高度
 * @param padw　补偿宽度
 * @param padh　补偿高度
 * @return　参数长度
 */
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

/**
 * softmax
 * @param pbox  feature map
 */
void softmax(const struct pBox *pbox) {
    if (pbox->pdata == NULL) {
        cout << "the softmax's pdata is NULL , Please check !" << endl;
        return;
    }
    mydataFmt *p2D = pbox->pdata;
    mydataFmt *p3D = NULL;
    long mapSize = pbox->width * pbox->height;
    mydataFmt eleSum = 0;
    for (int row = 0; row < pbox->height; row++) {
        for (int col = 0; col < pbox->width; col++) {
            eleSum = 0;
            for (int channel = 0; channel < pbox->channel; channel++) {
                p3D = p2D + channel * mapSize;
                *p3D = exp(*p3D);
                eleSum += *p3D;
            }
            for (int channel = 0; channel < pbox->channel; channel++) {
                p3D = p2D + channel * mapSize;
                *p3D = (*p3D) / eleSum;
            }
            p2D++;
        }
    }
}

bool cmpScore(struct orderScore lsh, struct orderScore rsh) {
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}

/**
 * 非极大值抑制
 * @param boundingBox_
 * @param bboxScore_
 * @param overlap_threshold
 * @param modelname
 */
void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const mydataFmt overlap_threshold,
         string modelname) {
    if (boundingBox_.empty()) {
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while (bboxScore_.size() > 0) {
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if (order < 0)continue;
        heros.push_back(order);
        boundingBox_.at(order).exist = false;//delete it

        for (int num = 0; num < boundingBox_.size(); num++) {
            if (boundingBox_.at(num).exist) {
                //the iou
                maxX = (boundingBox_.at(num).x1 > boundingBox_.at(order).x1) ? boundingBox_.at(num).x1
                                                                             : boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1 > boundingBox_.at(order).y1) ? boundingBox_.at(num).y1
                                                                             : boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2 < boundingBox_.at(order).x2) ? boundingBox_.at(num).x2
                                                                             : boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2 < boundingBox_.at(order).y2) ? boundingBox_.at(num).y2
                                                                             : boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse
                maxX = ((minX - maxX + 1) > 0) ? (minX - maxX + 1) : 0;
                maxY = ((minY - maxY + 1) > 0) ? (minY - maxY + 1) : 0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if (!modelname.compare("Union"))
                    IOU = IOU / (boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if (!modelname.compare("Min")) {
                    IOU = IOU /
                          ((boundingBox_.at(num).area < boundingBox_.at(order).area) ? boundingBox_.at(num).area
                                                                                     : boundingBox_.at(
                                          order).area);
                }
                if (IOU > overlap_threshold) {
                    boundingBox_.at(num).exist = false;
                    for (vector<orderScore>::iterator it = bboxScore_.begin(); it != bboxScore_.end(); it++) {
                        if ((*it).oriOrder == num) {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }
    for (int i = 0; i < heros.size(); i++)
        boundingBox_.at(heros.at(i)).exist = true;
}

void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width) {
    if (vecBbox.empty()) {
        cout << "Bbox is empty!!" << endl;
        return;
    }
    float bbw = 0, bbh = 0, maxSide = 0;
    float h = 0, w = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    for (vector<struct Bbox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++) {
        if ((*it).exist) {
            bbh = (*it).x2 - (*it).x1 + 1;
            bbw = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[1] * bbh;
            y1 = (*it).y1 + (*it).regreCoord[0] * bbw;
            x2 = (*it).x2 + (*it).regreCoord[3] * bbh;
            y2 = (*it).y2 + (*it).regreCoord[2] * bbw;

            h = x2 - x1 + 1;
            w = y2 - y1 + 1;

            maxSide = (h > w) ? h : w;
            x1 = x1 + h * 0.5 - maxSide * 0.5;
            y1 = y1 + w * 0.5 - maxSide * 0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if ((*it).x1 < 0)(*it).x1 = 0;
            if ((*it).y1 < 0)(*it).y1 = 0;
            if ((*it).x2 > height)(*it).x2 = height - 1;
            if ((*it).y2 > width)(*it).y2 = width - 1;

            it->area = (it->x2 - it->x1) * (it->y2 - it->y1);
        }
    }
}

/**
 * 残差融合初始化
 * @param output 输出feature map
 * @param c1 输入feature map
 * @param c2 输入feature map
 * @param c3 输入feature map
 * @param c4 输入feature map
 */
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

/**
 * 残差网络融合
 * @param output 输出feature map
 * @param c1 输入feature map
 * @param c2 输入feature map
 * @param c3 输入feature map
 * @param c4 输入feature map
 */
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

/**
 * 残差网络做多次按比例相加初始化
 * @param inpbox 输入feature map
 * @param temppbox 输入feature map
 * @param outpBox 输出feature map
 */
void mulandaddInit(const pBox *inpbox, const pBox *temppbox, pBox *outpBox) {
    if (!((inpbox->width == temppbox->width) && (inpbox->height == temppbox->height) &&
          (inpbox->channel == temppbox->channel)))
        cout << "the mulandaddInit is failed!!" << endl;
    outpBox->channel = temppbox->channel;
    outpBox->width = temppbox->width;
    outpBox->height = temppbox->height;
    outpBox->pdata = (mydataFmt *) malloc(outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
    if (outpBox->pdata == NULL)cout << "the mulandaddInit is failed!!" << endl;
    memset(outpBox->pdata, 0, outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
}

/**
 * 残差网络做多次按比例相加
 * @param inpbox 输入feature map
 * @param temppbox 输入feature map
 * @param outpBox 输出feature map
 * @param scale 比例系数
 */
void mulandadd(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale) {
    mydataFmt *ip = inpbox->pdata;
    mydataFmt *tp = temppbox->pdata;
    mydataFmt *op = outpBox->pdata;
    long dis = inpbox->width * inpbox->height * inpbox->channel;
    for (long i = 0; i < dis; i++) {
        op[i] = ip[i] + tp[i] * scale;
    }
}


/**
 * BN初始化
 * @param beta beta
 * @param mean 平均值
 * @param var 方差
 * @param width 参数个数
 */
void BatchNormInit(struct BN *beta, struct BN *mean, struct BN *var, int width) {
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

/**
 * BN实现
 * @param pbox　输入feature map
 * @param beta beta
 * @param mean 平均值
 * @param var 方差
 */
void BatchNorm(struct pBox *pbox, struct BN *beta, struct BN *mean, struct BN *var) {
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