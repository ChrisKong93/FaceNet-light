#include "network.h"
#include "facenet.h"
#include <time.h>

/**
 * 图片缩小
 * @param src 输入图片
 * @return 返回图片
 */
Mat RS(Mat &src) {
    int w = src.cols;
    int h = src.rows;
    int wtemp, htemp;
    Mat dst;
    cout << w << "\t" << h << endl;
    float threshold = 300.0;
    if (h > threshold) {
        wtemp = (int) (threshold / h * w);
        htemp = threshold;
        dst = Mat::zeros(htemp, wtemp, CV_8UC3); //我要转化为htemp*wtemp大小的
        resize(src, dst, dst.size());
    }
    cout << wtemp << "\t" << htemp << endl;
    cout << "-------------------" << endl;
    return dst;
}

/**
 * 对比两个人的emb值，计算空间欧氏距离
 * @param lineArray0 第一个人的emb值
 * @param lineArray1 第二个人的emb值
 * @return
 */
float compare(vector<mydataFmt> &lineArray0, vector<mydataFmt> &lineArray1) {
    mydataFmt sum = 0;
    for (int i = 0; i < Num; ++i) {
//        cout << lineArray0[i] << "===" << lineArray1[i] << endl;
        mydataFmt sub = lineArray0[i] - lineArray1[i];
        mydataFmt square = pow(sub, 2);
        sum += square;
    }
    mydataFmt result = sqrt(sum);
    return result;
}


/**
 * 执行单次单人的facenet网络
 * @param image 输入图片
 * @param vecRect 人脸框
 * @param n emb值
 */
void test_facenet(Mat &image, vector<mydataFmt> &n) {
    Mat fourthImage;
    resize(image, fourthImage, Size(160, 160), 0, 0, cv::INTER_LINEAR);
    facenet ggg;
//        mydataFmt *o = new mydataFmt[Num];
//    vector<mydataFmt> n;
//    vector<vector<mydataFmt>> o;
    ggg.run(fourthImage, n, 0);
}

void test() {
    Mat image = imread("../1.jpg");
//        Mat image = imread("../2.png");
    Mat Image;
    resize(image, Image, Size(160, 160), 0, 0, cv::INTER_LINEAR);
    facenet ggg;
    vector<mydataFmt> o;
    ggg.run(Image, o, 0);
    imshow("result", Image);
    imwrite("../result.jpg", Image);

    for (int i = 0; i < Num; ++i) {
        cout << o[i] << endl;
    }

    waitKey(0);
    image.release();
}


/**
 * 对比两张图两个人的emb
 */
void compareperson() {
    Mat image0 = imread("../1.jpg");
    Mat image1 = imread("../9.jpg");
//    image0 = RS(image0);
//    image1 = RS(image1);

    clock_t start;
    start = clock();
    vector<mydataFmt> n0, n1;

    test_facenet(image0, n0);
    test_facenet(image1, n1);


    float result = compare(n0, n1);
    cout << "-------------------" << endl;
    cout << result << endl;
    if (result < 0.45)
        cout << "Probably the same person" << endl;
    else
        cout << "Probably not the same person" << endl;

    imshow("result0", image0);
//    resizeWindow("result0", w0, h0); //创建一个固定值大小的窗口
    imwrite("../test_img/result0.jpg", image0);
    imshow("result1", image1);
    imwrite("../test_img/result1.jpg", image1);
    start = clock() - start;
    //    cout<<"time is  "<<start/10e3<<endl;
    cout << "time is " << (double) start / CLOCKS_PER_SEC * 1000 << "ms" << endl;
    waitKey(5000);
    image0.release();
    image1.release();
}

int main() {
//    test();
    compareperson();
    return 0;
}
