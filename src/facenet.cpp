//
// Created by ChrisKong on 2019/10/17.
//

#include "facenet.h"

facenet::facenet() {

}

facenet::~facenet() {

}

void facenet::printData(pBox *in) {
    for (long i = 0; i < in->height * in->width * in->channel; ++i) {
        printf("%f\n", in->pdata[i]);
    }
    cout << "printData" << endl;
}

void facenet::run(Mat &image, mydataFmt *o, int count) {
    cout << "=====This is No." + to_string(count) + " Picture=====" << endl;
    pBox *output = new pBox;
    pBox *input;
//    prewhiten(image);
    Stem(image, output);
//    printData(output);
//    return;
    cout << "Stem Finally" << endl;
    input = output;
    output = new pBox;
    for (int i = 0; i < 5; ++i) {
//        model_128/block35_1_list.txt
        string filepath = "../model_" + to_string(Num) + "/block35_" + to_string((i + 1)) + "_list.txt";
        Inception_resnet_A(input, output, filepath, 0.17);
        input = output;
        output = new pBox;
    }
    cout << "Inception_resnet_A Finally" << endl;
    Reduction_A(input, output);
    cout << "Reduction_A Finally" << endl;
    input = output;
//    freepBox(output);
    output = new pBox;
    for (int j = 0; j < 10; ++j) {
//        model_128/block17_1_list.txt
        string filepath = "../model_" + to_string(Num) + "/block17_" + to_string((j + 1)) + "_list.txt";
        Inception_resnet_B(input, output, filepath, 0.1);
        input = output;
        output = new pBox;
    }
    cout << "Inception_resnet_B Finally" << endl;
    Reduction_B(input, output);
    cout << "Reduciotn_B Finally" << endl;
    input = output;
//    freepBox(output);
    output = new pBox;
    for (int k = 0; k < 5; ++k) {
//        model_128/block8_1_list.txt
        string filepath = "../model_" + to_string(Num) + "/block8_" + to_string((k + 1)) + "_list.txt";
//        cout << filepath << endl;
        Inception_resnet_C(input, output, filepath, 0.2);
        input = output;
//        freepBox(output);
        output = new pBox;
    }
    cout << "Inception_resnet_C Finally" << endl;
    Inception_resnet_C_None(input, output, "../model_" + to_string(Num) + "/Block8_list.txt");
    cout << "Inception_resnet_C_None Finally" << endl;
    input = output;
//    freepBox(output);
    output = new pBox;
    AveragePooling(input, output);
    cout << "AveragePooling Finally" << endl;
    input = output;
//    freepBox(output);
    output = new pBox;
    Flatten(input, output);
    cout << "Flatten Finally" << endl;
    input = output;
    output = new pBox;
    fully_connect(input, output, "../model_" + to_string(Num) + "/Bottleneck_list.txt");
    cout << "Fully_Connect Finally" << endl;
    mydataFmt sq = 0, sum = 0;
    for (int i = 0; i < Num; ++i) {
        sq = pow(output->pdata[i], 2);
        sum += sq;
    }
    mydataFmt divisor = 0;
    if (sum < 1e-10) {
        divisor = sqrt(1e-10);
    } else {
        divisor = sqrt(sum);
    }
    for (int j = 0; j < Num; ++j) {
        o[j] = output->pdata[j] / divisor;
    }
//    memcpy(o, output->pdata, Num * sizeof(mydataFmt));
    freepBox(output);
}

void facenet::Stem(Mat &image, pBox *output) {
    pBox *rgb = new pBox;
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;

    struct pBox *pooling1_out = new pBox;

    struct pRelu *prelu_gmma1 = new pRelu;
    struct pRelu *prelu_gmma2 = new pRelu;
    struct pRelu *prelu_gmma3 = new pRelu;
    struct pRelu *prelu_gmma4 = new pRelu;
    struct pRelu *prelu_gmma5 = new pRelu;
    struct pRelu *prelu_gmma6 = new pRelu;


    long conv1 = initConvAndFc(conv1_wb, 32, 3, 3, 2, 0);
    initpRelu(prelu_gmma1, 32);
    long conv2 = initConvAndFc(conv2_wb, 32, 32, 3, 1, 0);
    initpRelu(prelu_gmma2, 32);
    long conv3 = initConvAndFc(conv3_wb, 64, 32, 3, 1, 1);
    initpRelu(prelu_gmma3, 64);
    long conv4 = initConvAndFc(conv4_wb, 80, 64, 1, 1, 0);
    initpRelu(prelu_gmma4, 80);
    long conv5 = initConvAndFc(conv5_wb, 192, 80, 3, 1, 0);
    initpRelu(prelu_gmma5, 192);
    long conv6 = initConvAndFc(conv6_wb, 256, 192, 3, 2, 0);
    initpRelu(prelu_gmma6, 256);
    long dataNumber[18] = {conv1, 0, 0, conv2, 0, 0, conv3, 0, 0, conv4, 0, 0, conv5, 0, 0, conv6, 0,
                           0};

    mydataFmt *pointTeam[18] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                            conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                            conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                            conv4_wb->pdata, conv4_wb->pbias, prelu_gmma4->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias, prelu_gmma5->pdata, \
                            conv6_wb->pdata, conv6_wb->pbias, prelu_gmma6->pdata,};
    string filename = "../model_" + to_string(Num) + "/stem_list.txt";
    readData(filename, dataNumber, pointTeam);



//    if (firstFlag) {
    image2MatrixInit(image, rgb);
    image2Matrix(image, rgb, 1);

    convolutionInit(conv1_wb, rgb, conv1_out);
    //conv1 149 x 149 x 32
    convolution(conv1_wb, rgb, conv1_out);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);
    convolutionInit(conv2_wb, conv1_out, conv2_out);
    //conv2 147 x 147 x 32
    convolution(conv2_wb, conv1_out, conv2_out);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 147 x 147 x 64
    convolution(conv3_wb, conv2_out, conv3_out);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);

    maxPoolingInit(conv3_out, pooling1_out, 3, 2);
    //maxPooling 73 x 73 x 64
    maxPooling(conv3_out, pooling1_out, 3, 2);

    convolutionInit(conv4_wb, pooling1_out, conv4_out);
    //conv4 73 x 73 x 80
    convolution(conv4_wb, pooling1_out, conv4_out);
    prelu(conv4_out, conv4_wb->pbias, prelu_gmma4->pdata);

    convolutionInit(conv5_wb, conv4_out, conv5_out);
    //conv5 71 x 71 x 192
    convolution(conv5_wb, conv4_out, conv5_out);
    prelu(conv5_out, conv5_wb->pbias, prelu_gmma5->pdata);


    convolutionInit(conv6_wb, conv5_out, output);
    //conv6 35 x 35 x 256
    convolution(conv6_wb, conv5_out, output);
    prelu(output, conv6_wb->pbias, prelu_gmma6->pdata);
//        firstFlag = false;
//    }

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(pooling1_out);

    freepBox(rgb);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);

    freepRelu(prelu_gmma1);
    freepRelu(prelu_gmma2);
    freepRelu(prelu_gmma3);
    freepRelu(prelu_gmma4);
    freepRelu(prelu_gmma5);
    freepRelu(prelu_gmma6);
}

void facenet::Inception_resnet_A(pBox *input, pBox *output, string filepath, float scale) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;
    pBox *conv7_out = new pBox;
    pBox *conv8_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;
    struct Weight *conv7_wb = new Weight;
    struct Weight *conv8_wb = new Weight;

    struct pRelu *prelu_gmma1 = new pRelu;
    struct pRelu *prelu_gmma2 = new pRelu;
    struct pRelu *prelu_gmma3 = new pRelu;
    struct pRelu *prelu_gmma4 = new pRelu;
    struct pRelu *prelu_gmma5 = new pRelu;
    struct pRelu *prelu_gmma6 = new pRelu;
    struct pRelu *prelu_gmma8 = new pRelu;

    long conv1 = initConvAndFc(conv1_wb, 32, 256, 1, 1, 0);
    initpRelu(prelu_gmma1, 32);

    long conv2 = initConvAndFc(conv2_wb, 32, 256, 1, 1, 0);
    initpRelu(prelu_gmma2, 32);
    long conv3 = initConvAndFc(conv3_wb, 32, 32, 3, 1, 1);
    initpRelu(prelu_gmma3, 32);

    long conv4 = initConvAndFc(conv4_wb, 32, 256, 1, 1, 0);
    initpRelu(prelu_gmma4, 32);
    long conv5 = initConvAndFc(conv5_wb, 32, 32, 3, 1, 1);
    initpRelu(prelu_gmma5, 32);
    long conv6 = initConvAndFc(conv6_wb, 32, 32, 3, 1, 1);
    initpRelu(prelu_gmma6, 32);

    long conv7 = initConvAndFc(conv7_wb, 256, 96, 1, 1, 0);

    long conv8 = initConvAndFc(conv8_wb, 256, 0, 0, 0, 0);
    initpRelu(prelu_gmma8, 256);

    long dataNumber[23] = {conv1, 0, 0, conv2, 0, 0, conv3, 0, 0, conv4, 0, 0, conv5, 0, 0, conv6, 0,
                           0, conv7, 256, conv8, 0, 0};

    mydataFmt *pointTeam[23] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                            conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                            conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                            conv4_wb->pdata, conv4_wb->pbias, prelu_gmma4->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias, prelu_gmma5->pdata, \
                            conv6_wb->pdata, conv6_wb->pbias, prelu_gmma6->pdata, \
                            conv7_wb->pdata, conv7_wb->pbias, \
                            conv8_wb->pdata, conv8_wb->pbias, prelu_gmma8->pdata};

    readData(filepath, dataNumber, pointTeam);


    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 35 x 35 x 32
    convolution(conv1_wb, input, conv1_out);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 35 x 35 x 32
    convolution(conv2_wb, input, conv2_out);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);
    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 35 x 35 x 32
    convolution(conv3_wb, conv2_out, conv3_out);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);

    convolutionInit(conv4_wb, input, conv4_out);
    //conv4 35 x 35 x 32
    convolution(conv4_wb, input, conv4_out);
    prelu(conv4_out, conv4_wb->pbias, prelu_gmma4->pdata);
    convolutionInit(conv5_wb, conv4_out, conv5_out);
    //conv5 35 x 35 x 32
    convolution(conv5_wb, conv4_out, conv5_out);
    prelu(conv5_out, conv5_wb->pbias, prelu_gmma5->pdata);
    convolutionInit(conv6_wb, conv5_out, conv6_out);
    //conv6 35 x 35 x 32
    convolution(conv6_wb, conv5_out, conv6_out);
    prelu(conv6_out, conv6_wb->pbias, prelu_gmma6->pdata);

    conv_mergeInit(conv7_out, conv1_out, conv3_out, conv6_out);
    //35 × 35 × 96
    conv_merge(conv7_out, conv1_out, conv3_out, conv6_out);

    convolutionInit(conv7_wb, conv7_out, conv8_out);
    //35*35*256
    convolution(conv7_wb, conv7_out, conv8_out);
    addbias(conv8_out, conv7_wb->pbias);

    mulandaddInit(input, conv8_out, output, scale);
    mulandadd(input, conv8_out, output, scale);
    prelu(output, conv8_wb->pbias, prelu_gmma8->pdata);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);
    freepBox(conv7_out);
    freepBox(conv8_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);
    freeWeight(conv7_wb);
    freeWeight(conv8_wb);

    freepRelu(prelu_gmma1);
    freepRelu(prelu_gmma2);
    freepRelu(prelu_gmma3);
    freepRelu(prelu_gmma4);
    freepRelu(prelu_gmma5);
    freepRelu(prelu_gmma6);
    freepRelu(prelu_gmma8);
}

void facenet::Reduction_A(pBox *input, pBox *output) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;

    struct pBox *pooling1_out = new pBox;

    struct pRelu *prelu_gmma1 = new pRelu;
    struct pRelu *prelu_gmma2 = new pRelu;
    struct pRelu *prelu_gmma3 = new pRelu;
    struct pRelu *prelu_gmma4 = new pRelu;

    long conv1 = initConvAndFc(conv1_wb, 384, 256, 3, 2, 0);
    initpRelu(prelu_gmma1, 384);
    long conv2 = initConvAndFc(conv2_wb, 192, 256, 1, 1, 0);
    initpRelu(prelu_gmma2, 192);
    long conv3 = initConvAndFc(conv3_wb, 192, 192, 3, 1, 0);
    initpRelu(prelu_gmma3, 192);
    long conv4 = initConvAndFc(conv4_wb, 256, 192, 3, 2, 0);
    initpRelu(prelu_gmma4, 256);
    long dataNumber[12] = {conv1, 0, 0, conv2, 0, 0, conv3, 0, 0, conv4, 0, 0};

    mydataFmt *pointTeam[12] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                            conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                            conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                            conv4_wb->pdata, conv4_wb->pbias, prelu_gmma4->pdata};
    string filename = "../model_" + to_string(Num) + "/Mixed_6a_list.txt";
    readData(filename, dataNumber, pointTeam);

    maxPoolingInit(input, pooling1_out, 3, 2);
    // 17*17*256
    maxPooling(input, pooling1_out, 3, 2);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 17 x 17 x 384
    convolution(conv1_wb, input, conv1_out);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 35 x 35 x 192
    convolution(conv2_wb, input, conv2_out);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 35 x 35 x 192
    convolution(conv3_wb, conv2_out, conv3_out);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 17 x 17 x 256
    convolution(conv4_wb, conv3_out, conv4_out);
    prelu(conv4_out, conv4_wb->pbias, prelu_gmma4->pdata);
    conv_mergeInit(output, pooling1_out, conv1_out, conv4_out);
    //17×17×896
    conv_merge(output, pooling1_out, conv1_out, conv4_out);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);

    freepBox(pooling1_out);

    freepRelu(prelu_gmma1);
    freepRelu(prelu_gmma2);
    freepRelu(prelu_gmma3);
    freepRelu(prelu_gmma4);
}

void facenet::Inception_resnet_B(pBox *input, pBox *output, string filepath, float scale) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;

    struct pRelu *prelu_gmma1 = new pRelu;
    struct pRelu *prelu_gmma2 = new pRelu;
    struct pRelu *prelu_gmma3 = new pRelu;
    struct pRelu *prelu_gmma4 = new pRelu;
    struct pRelu *prelu_gmma6 = new pRelu;

    long conv1 = initConvAndFc(conv1_wb, 128, 896, 1, 1, 0);
    initpRelu(prelu_gmma1, 128);
    long conv2 = initConvAndFc(conv2_wb, 128, 896, 1, 1, 0);
    initpRelu(prelu_gmma2, 128);
    long conv3 = initConvAndFc(conv3_wb, 128, 128, 0, 1, -1, 1, 7, 0, 3);//[1,7]
    initpRelu(prelu_gmma3, 128);
    long conv4 = initConvAndFc(conv4_wb, 128, 128, 0, 1, -1, 7, 1, 3, 0);//[7,1]
    initpRelu(prelu_gmma4, 128);

    long conv5 = initConvAndFc(conv5_wb, 896, 256, 1, 1, 0);

    long conv6 = initConvAndFc(conv6_wb, 896, 0, 0, 0, 0);
    initpRelu(prelu_gmma6, 896);

    long dataNumber[17] = {conv1, 0, 0, conv2, 0, 0, conv3, 0, 0, conv4, 0, 0, conv5, 896, conv6, 0,
                           0};

    mydataFmt *pointTeam[17] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                            conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                            conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                            conv4_wb->pdata, conv4_wb->pbias, prelu_gmma4->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias, \
                            conv6_wb->pdata, conv6_wb->pbias, prelu_gmma6->pdata};


    readData(filepath, dataNumber, pointTeam);


    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 17*17*128
    convolution(conv1_wb, input, conv1_out);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 17*17*128
    convolution(conv2_wb, input, conv2_out);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 17*17*128
    convolution(conv3_wb, conv2_out, conv3_out);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 17*17*128
    convolution(conv4_wb, conv3_out, conv4_out);
    prelu(conv4_out, conv4_wb->pbias, prelu_gmma4->pdata);

    conv_mergeInit(conv5_out, conv1_out, conv4_out);
    //17*17*256
    conv_merge(conv5_out, conv1_out, conv4_out);

    convolutionInit(conv5_wb, conv5_out, conv6_out);
    //conv5 17*17*896
    convolution(conv5_wb, conv5_out, conv6_out);
    addbias(conv6_out, conv5_wb->pbias);

    mulandaddInit(input, conv6_out, output, scale);
    mulandadd(input, conv6_out, output, scale);
    prelu(output, conv6_wb->pbias, prelu_gmma6->pdata);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);

    freepRelu(prelu_gmma1);
    freepRelu(prelu_gmma2);
    freepRelu(prelu_gmma3);
    freepRelu(prelu_gmma4);
//    freepRelu(prelu_gmma5);
    freepRelu(prelu_gmma6);
}

void facenet::Reduction_B(pBox *input, pBox *output) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;
    pBox *conv7_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;
    struct Weight *conv7_wb = new Weight;

    struct pBox *pooling1_out = new pBox;

    struct pRelu *prelu_gmma1 = new pRelu;
    struct pRelu *prelu_gmma2 = new pRelu;
    struct pRelu *prelu_gmma3 = new pRelu;
    struct pRelu *prelu_gmma4 = new pRelu;
    struct pRelu *prelu_gmma5 = new pRelu;
    struct pRelu *prelu_gmma6 = new pRelu;
    struct pRelu *prelu_gmma7 = new pRelu;

    long conv1 = initConvAndFc(conv1_wb, 256, 896, 1, 1, 0);
    initpRelu(prelu_gmma1, 256);
    long conv2 = initConvAndFc(conv2_wb, 384, 256, 3, 2, 0);
    initpRelu(prelu_gmma2, 384);

    long conv3 = initConvAndFc(conv3_wb, 256, 896, 1, 1, 0);
    initpRelu(prelu_gmma3, 256);
    long conv4 = initConvAndFc(conv4_wb, 256, 256, 3, 2, 0);
    initpRelu(prelu_gmma4, 256);

    long conv5 = initConvAndFc(conv5_wb, 256, 896, 1, 1, 0);
    initpRelu(prelu_gmma5, 256);
    long conv6 = initConvAndFc(conv6_wb, 256, 256, 3, 1, 1);
    initpRelu(prelu_gmma6, 256);
    long conv7 = initConvAndFc(conv7_wb, 256, 256, 3, 2, 0);
    initpRelu(prelu_gmma7, 256);

    long dataNumber[21] = {conv1, 0, 0, conv2, 0, 0, conv3, 0, 0, conv4, 0, 0, conv5, 0, 0, conv6,
                           0, 0, conv7, 0, 0};

    mydataFmt *pointTeam[21] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                            conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                            conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                            conv4_wb->pdata, conv4_wb->pbias, prelu_gmma4->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias, prelu_gmma5->pdata, \
                            conv6_wb->pdata, conv6_wb->pbias, prelu_gmma6->pdata, \
                            conv7_wb->pdata, conv7_wb->pbias, prelu_gmma7->pdata,};
    string filename = "../model_" + to_string(Num) + "/Mixed_7a_list.txt";
    readData(filename, dataNumber, pointTeam);


    maxPoolingInit(input, pooling1_out, 3, 2, 1);
    // 8*8*896
    maxPooling(input, pooling1_out, 3, 2);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 17 x 17 x 256
    convolution(conv1_wb, input, conv1_out);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);

    convolutionInit(conv2_wb, conv1_out, conv2_out);
    //conv2 8 x 8 x 384
    convolution(conv2_wb, conv1_out, conv2_out);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);

    convolutionInit(conv3_wb, input, conv3_out);
    //conv3 17 x 17 x 256
    convolution(conv3_wb, input, conv3_out);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 8 x 8 x 256
    convolution(conv4_wb, conv3_out, conv4_out);
    prelu(conv4_out, conv4_wb->pbias, prelu_gmma4->pdata);

    convolutionInit(conv5_wb, input, conv5_out);
    //conv5 17 x 17 x 256
    convolution(conv5_wb, input, conv5_out);
    prelu(conv5_out, conv5_wb->pbias, prelu_gmma5->pdata);

    convolutionInit(conv6_wb, conv5_out, conv6_out);
    //conv6 17 x 17 x 256
    convolution(conv6_wb, conv5_out, conv6_out);
    prelu(conv6_out, conv6_wb->pbias, prelu_gmma6->pdata);

    convolutionInit(conv7_wb, conv6_out, conv7_out);
    //conv6 8 x 8 x 256
    convolution(conv7_wb, conv6_out, conv7_out);
    prelu(conv7_out, conv7_wb->pbias, prelu_gmma7->pdata);

    conv_mergeInit(output, conv2_out, conv4_out, conv7_out, pooling1_out);
    //8*8*1792
    conv_merge(output, conv2_out, conv4_out, conv7_out, pooling1_out);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);
    freepBox(conv7_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);
    freeWeight(conv7_wb);

    freepBox(pooling1_out);

    freepRelu(prelu_gmma1);
    freepRelu(prelu_gmma2);
    freepRelu(prelu_gmma3);
    freepRelu(prelu_gmma4);
    freepRelu(prelu_gmma5);
    freepRelu(prelu_gmma6);
    freepRelu(prelu_gmma7);
}

void facenet::Inception_resnet_C(pBox *input, pBox *output, string filepath, float scale) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;
    struct Weight *conv6_wb = new Weight;

    struct pRelu *prelu_gmma1 = new pRelu;
    struct pRelu *prelu_gmma2 = new pRelu;
    struct pRelu *prelu_gmma3 = new pRelu;
    struct pRelu *prelu_gmma4 = new pRelu;
    struct pRelu *prelu_gmma6 = new pRelu;


    long conv1 = initConvAndFc(conv1_wb, 192, 1792, 1, 1, 0);
    initpRelu(prelu_gmma1, 192);
    long conv2 = initConvAndFc(conv2_wb, 192, 1792, 1, 1, 0);
    initpRelu(prelu_gmma2, 192);
    long conv3 = initConvAndFc(conv3_wb, 192, 192, 0, 1, -1, 1, 3, 0, 1);
    initpRelu(prelu_gmma3, 192);
    long conv4 = initConvAndFc(conv4_wb, 192, 192, 0, 1, -1, 3, 1, 1, 0);
    initpRelu(prelu_gmma4, 192);
    long conv5 = initConvAndFc(conv5_wb, 1792, 384, 1, 1, 0);

    long conv6 = initConvAndFc(conv6_wb, 1792, 0, 0, 0, 0);
    initpRelu(prelu_gmma6, 1792);

    long dataNumber[17] = {conv1, 0, 0, conv2, 0, 0, conv3, 0, 0, conv4, 0, 0, conv5, 1792, conv6, 0,
                           0};


    mydataFmt *pointTeam[17] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                            conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                            conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                            conv4_wb->pdata, conv4_wb->pbias, prelu_gmma4->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias, \
                            conv6_wb->pdata, conv6_wb->pbias, prelu_gmma6->pdata};

//    string filename = "../model_128/Repeat_2_list.txt";
//    int length = sizeof(dataNumber) / sizeof(*dataNumber);
    readData(filepath, dataNumber, pointTeam);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 8 x 8 x 192
    convolution(conv1_wb, input, conv1_out);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 8 x 8 x 192
    convolution(conv2_wb, input, conv2_out);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 8 x 8 x 192
    convolution(conv3_wb, conv2_out, conv3_out);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 8 x 8 x 192
    convolution(conv4_wb, conv3_out, conv4_out);
    prelu(conv4_out, conv4_wb->pbias, prelu_gmma4->pdata);

    conv_mergeInit(conv5_out, conv1_out, conv4_out);
    // 8*8*384
    conv_merge(conv5_out, conv1_out, conv4_out);

    convolutionInit(conv5_wb, conv5_out, conv6_out);
    //conv5 8 x 8 x 1792
    convolution(conv5_wb, conv5_out, conv6_out);
    addbias(conv6_out, conv5_wb->pbias);

    mulandaddInit(input, conv6_out, output, scale);
    mulandadd(input, conv6_out, output, scale);
    prelu(output, conv6_wb->pbias, prelu_gmma6->pdata);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);
    freeWeight(conv6_wb);

    freepRelu(prelu_gmma1);
    freepRelu(prelu_gmma2);
    freepRelu(prelu_gmma3);
    freepRelu(prelu_gmma4);
//    freepRelu(prelu_gmma5);
    freepRelu(prelu_gmma6);
}

void facenet::Inception_resnet_C_None(pBox *input, pBox *output, string filepath) {
    pBox *conv1_out = new pBox;
    pBox *conv2_out = new pBox;
    pBox *conv3_out = new pBox;
    pBox *conv4_out = new pBox;
    pBox *conv5_out = new pBox;
    pBox *conv6_out = new pBox;

    struct Weight *conv1_wb = new Weight;
    struct Weight *conv2_wb = new Weight;
    struct Weight *conv3_wb = new Weight;
    struct Weight *conv4_wb = new Weight;
    struct Weight *conv5_wb = new Weight;

    struct pRelu *prelu_gmma1 = new pRelu;
    struct pRelu *prelu_gmma2 = new pRelu;
    struct pRelu *prelu_gmma3 = new pRelu;
    struct pRelu *prelu_gmma4 = new pRelu;


    long conv1 = initConvAndFc(conv1_wb, 192, 1792, 1, 1, 0);
    initpRelu(prelu_gmma1, 192);
    long conv2 = initConvAndFc(conv2_wb, 192, 1792, 1, 1, 0);
    initpRelu(prelu_gmma2, 192);
    long conv3 = initConvAndFc(conv3_wb, 192, 192, 0, 1, -1, 1, 3, 0, 1);
    initpRelu(prelu_gmma3, 192);
    long conv4 = initConvAndFc(conv4_wb, 192, 192, 0, 1, -1, 3, 1, 1, 0);
    initpRelu(prelu_gmma4, 192);
    long conv5 = initConvAndFc(conv5_wb, 1792, 384, 1, 1, 0);

    long dataNumber[14] = {conv1, 0, 0, conv2, 0, 0, conv3, 0, 0, conv4, 0, 0, conv5, 1792};


    mydataFmt *pointTeam[14] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata, \
                            conv2_wb->pdata, conv2_wb->pbias, prelu_gmma2->pdata, \
                            conv3_wb->pdata, conv3_wb->pbias, prelu_gmma3->pdata, \
                            conv4_wb->pdata, conv4_wb->pbias, prelu_gmma4->pdata, \
                            conv5_wb->pdata, conv5_wb->pbias};

//    string filename = "../model_128/Repeat_2_list.txt";
//    int length = sizeof(dataNumber) / sizeof(*dataNumber);
    readData(filepath, dataNumber, pointTeam);

    convolutionInit(conv1_wb, input, conv1_out);
    //conv1 8 x 8 x 192
    convolution(conv1_wb, input, conv1_out);
    prelu(conv1_out, conv1_wb->pbias, prelu_gmma1->pdata);

    convolutionInit(conv2_wb, input, conv2_out);
    //conv2 8 x 8 x 192
    convolution(conv2_wb, input, conv2_out);
    prelu(conv2_out, conv2_wb->pbias, prelu_gmma2->pdata);

    convolutionInit(conv3_wb, conv2_out, conv3_out);
    //conv3 8 x 8 x 192
    convolution(conv3_wb, conv2_out, conv3_out);
    prelu(conv3_out, conv3_wb->pbias, prelu_gmma3->pdata);

    convolutionInit(conv4_wb, conv3_out, conv4_out);
    //conv4 8 x 8 x 192
    convolution(conv4_wb, conv3_out, conv4_out);
    prelu(conv4_out, conv4_wb->pbias, prelu_gmma4->pdata);

    conv_mergeInit(conv5_out, conv1_out, conv4_out);
    // 8*8*384
    conv_merge(conv5_out, conv1_out, conv4_out);

    convolutionInit(conv5_wb, conv5_out, conv6_out);
    //conv5 8 x 8 x 1792
    convolution(conv5_wb, conv5_out, conv6_out);
    addbias(conv6_out, conv5_wb->pbias);

    mulandaddInit(input, conv6_out, output, 1);
    mulandadd(input, conv6_out, output);

    freepBox(conv1_out);
    freepBox(conv2_out);
    freepBox(conv3_out);
    freepBox(conv4_out);
    freepBox(conv5_out);
    freepBox(conv6_out);

    freeWeight(conv1_wb);
    freeWeight(conv2_wb);
    freeWeight(conv3_wb);
    freeWeight(conv4_wb);
    freeWeight(conv5_wb);

    freepRelu(prelu_gmma1);
    freepRelu(prelu_gmma2);
    freepRelu(prelu_gmma3);
    freepRelu(prelu_gmma4);
}

void facenet::AveragePooling(pBox *input, pBox *output) {
//    cout << "size:" << input->height << endl;
    avePoolingInit(input, output, input->height, 2);
    avePooling(input, output, input->height, 2);
}

void facenet::Flatten(pBox *input, pBox *output) {
    output->width = input->channel;
    output->height = 1;
    output->channel = 1;
    output->pdata = (mydataFmt *) malloc(output->channel * output->width * output->height * sizeof(mydataFmt));
    if (output->pdata == NULL)cout << "the maxPoolingInit is failed!!" << endl;
    memcpy(output->pdata, input->pdata, output->channel * output->width * output->height * sizeof(mydataFmt));
}

//参数还未设置
void facenet::fully_connect(pBox *input, pBox *output, string filepath) {
    struct Weight *conv1_wb = new Weight;
    struct pRelu *prelu_gmma1 = new pRelu;
    long conv1 = initConvAndFc(conv1_wb, Num, 1792, input->height, 1, 0);
    initpRelu(prelu_gmma1, Num);
    long dataNumber[3] = {conv1, 0, 0};

//    cout << to_string(sum) << endl;
    mydataFmt *pointTeam[3] = {conv1_wb->pdata, conv1_wb->pbias, prelu_gmma1->pdata};
//    string filename = "../model_128/Bottleneck_list.txt";
//    int length = sizeof(dataNumber) / sizeof(*dataNumber);
    readData(filepath, dataNumber, pointTeam);

    fullconnectInit(conv1_wb, output);

    //conv1 8 x 8 x 192
    fullconnect(conv1_wb, input, output);
//    prelu(output, conv1_wb->pbias, prelu_gmma1->pdata);

    freeWeight(conv1_wb);
    freepRelu(prelu_gmma1);
}

void facenet::conv_mergeInit(pBox *output, pBox *c1, pBox *c2, pBox *c3, pBox *c4) {
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
    } else { cout << "conv_mergeInit" << endl; }
    output->pdata = (mydataFmt *) malloc(output->width * output->height * output->channel * sizeof(mydataFmt));
    if (output->pdata == NULL)cout << "the conv_mergeInit is failed!!" << endl;
    memset(output->pdata, 0, output->width * output->height * output->channel * sizeof(mydataFmt));
}

void facenet::conv_merge(pBox *output, pBox *c1, pBox *c2, pBox *c3, pBox *c4) {
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
//    cout << "output->pdata:" << *(output->pdata) << endl;
}

void facenet::mulandaddInit(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale) {
    outpBox->channel = temppbox->channel;
    outpBox->width = temppbox->width;
    outpBox->height = temppbox->height;
    outpBox->pdata = (mydataFmt *) malloc(outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
    if (outpBox->pdata == NULL)cout << "the mulandaddInit is failed!!" << endl;
    memset(outpBox->pdata, 0, outpBox->width * outpBox->height * outpBox->channel * sizeof(mydataFmt));
}

void facenet::mulandadd(const pBox *inpbox, const pBox *temppbox, pBox *outpBox, float scale) {
    mydataFmt *ip = inpbox->pdata;
    mydataFmt *tp = temppbox->pdata;
    mydataFmt *op = outpBox->pdata;
    long dis = inpbox->width * inpbox->height * inpbox->channel;
    for (long i = 0; i < dis; i++) {
        op[i] = ip[i] + tp[i] * scale;
    }
}