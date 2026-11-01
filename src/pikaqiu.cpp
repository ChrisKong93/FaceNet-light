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
    int wtemp = w, htemp = h;
    Mat dst = src;
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
 * 对比两个人的emb值，计算余弦相似度
 * 取值范围 [-1, 1]，越接近 1 越相似
 * @param lineArray0 第一个人的emb值
 * @param lineArray1 第二个人的emb值
 * @return 余弦相似度
 */
float cosineSimilarity(vector<mydataFmt> &lineArray0, vector<mydataFmt> &lineArray1) {
    mydataFmt dot = 0, norm0 = 0, norm1 = 0;
    for (int i = 0; i < Num; ++i) {
        dot += lineArray0[i] * lineArray1[i];
        norm0 += lineArray0[i] * lineArray0[i];
        norm1 += lineArray1[i] * lineArray1[i];
    }
    mydataFmt denom = sqrt(norm0) * sqrt(norm1);
    if (denom < 1e-10) {
        denom = 1e-10;
    }
    return dot / denom;
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
    float cosSim = cosineSimilarity(n0, n1);
    cout << "-------------------" << endl;
    cout << "欧氏距离: " << result << endl;
    cout << "余弦相似度: " << cosSim << endl;
    // 512维模型（L2归一化后）常用阈值大致在 0.8~1.2 附近，具体以你自己的正负样本对为准
    if (result < 1.0)
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
    waitKey(0);
    image0.release();
    image1.release();
}

int main() {
//    test();
    compareperson();
    return 0;
}
