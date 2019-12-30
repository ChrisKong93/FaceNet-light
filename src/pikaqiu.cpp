#include "network.h"
#include "facenet.h"
#include <time.h>

int main() {
    int b = 0;
    if (b == 0) {
        Mat image = imread("../1.jpg");
//        Mat image = imread("../2.png");
        Mat Image;
        resize(image, Image, Size(299, 299), 0, 0, cv::INTER_LINEAR);
        facenet ggg;
        mydataFmt *o = new mydataFmt[Num];
        ggg.run(Image, o, 0);
//        imshow("result", Image);
        imwrite("../result.jpg", Image);

        for (int i = 0; i < Num; ++i) {
            cout << o[i] << endl;
        }

        waitKey(0);
        image.release();
    } else {
        Mat image;
        VideoCapture cap(0);
        if (!cap.isOpened())
            cout << "fail to open!" << endl;
        cap >> image;
        if (!image.data) {
            cout << "读取视频失败" << endl;
            return -1;
        }

        clock_t start;
        int stop = 1200;
        //while (stop--) {
        while (true) {
            start = clock();
            cap >> image;
            resize(image, image, Size(299, 299), 0, 0, cv::INTER_LINEAR);
            facenet ggg;
            mydataFmt *o = new mydataFmt[Num];
            ggg.run(image, o, 0);
            imshow("result", image);
            if (waitKey(1) >= 0) break;
            start = clock() - start;
            cout << "time is " << (double) start / CLOCKS_PER_SEC * 1000 << "ms" << endl;
        }
        waitKey(0);
        image.release();
    }
    return 0;
}
