#include "network.h"
#include "facenet.h"
#include <time.h>

int main() {
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

    return 0;
}
