# FaceNet-light

使用 C++ 实现的 FaceNet 人脸识别推理，**不依赖 TensorFlow**，仅使用 OpenCV 读取图片与加载权重，卷积、BatchNorm、池化、全连接等算子均为纯 C++ 手写实现。

模型参数提取自 FaceNet 官方发布的两个预训练模型：

| 模型 | embedding 维度 | 来源 |
| ---- | ---- | ---- |
| 20180402-114759 | 512 | [FaceNet 官方 2018 版](https://github.com/davidsandberg/facenet) |
| 20170512-110547 | 128 | [FaceNet 官方 2017 版](https://github.com/davidsandberg/facenet) |

## 网络结构

Backbone 为 Inception-ResNet-v1，整体流程：

```
输入 160x160x3
  → Stem
  → Inception-ResNet-A (block35) x5   (残差缩放 scale = 0.17)
  → Reduction-A
  → Inception-ResNet-B (block17) x10  (残差缩放 scale = 0.1)
  → Reduction-B
  → Inception-ResNet-C (block8) x5    (残差缩放 scale = 0.2)
  → Block8 (不带缩放)
  → AveragePooling
  → Fully Connected (Bottleneck)
  → L2 归一化
  → N 维 embedding
```

## 目录结构

```
.
├── src/
│   ├── pikaqiu.cpp    # 程序入口：加载两张图片并对比人脸
│   ├── facenet.cpp/h  # FaceNet 网络搭建（Stem、各 Inception-ResNet 块、全连接）
│   ├── network.cpp/h  # 基础算子：卷积、BatchNorm、Relu、池化、全连接、图片预处理
│   └── pBox.h/cpp     # 数据结构（feature map、卷积权重、BN 参数）与内存释放
├── model_512/         # 512 维模型权重（txt 格式，见下方说明）
├── model_128/         # 128 维模型权重（txt 格式，见下方说明）
├── 1.jpg / 9.jpg      # demo 用的对比人脸图片
└── CMakeLists.txt
```

## 编译

依赖：CMake ≥ 3.5、C++11 编译器、OpenCV（推荐 3.x/4.x）。

```bash
cd FaceNet-light
mkdir -p build && cd build
cmake ..
make
```

## 运行

```bash
cd build
./main
```

注意：代码中以相对路径 `../1.jpg`、`../9.jpg` 读取图片，以 `../model_XXX/` 加载权重，
因此**必须在项目根目录的下一级目录（如 `build/`）中运行**，否则会找不到文件。

demo 逻辑（`src/pikaqiu.cpp` 的 `compareperson()`）：

1. 分别将 `1.jpg` 和 `9.jpg` 缩放至 160x160，送入 FaceNet 得到两组 embedding；
2. 计算两个 embedding 的**欧氏距离**与**余弦相似度**并打印；
3. 欧氏距离小于 **1.0** 判定为同一人（阈值可按自己的正负样本对调整）；
4. 两张输入图保存到 `test_img/`。

## 模型文件

权重以纯文本格式按层存放在 `model_512/`（stem、block35、block17、block8、Bottleneck 等每个子块一个 `*_list.txt` 文件），推理时按层加载。

由于体积原因，`model_128/`、`model_512/` 未纳入版本管理（见 `.gitignore`），需要自行准备并放入对应目录；也可从官方 TensorFlow 模型提取，`readData()` 的文件格式为每行一个权重值、按层参数顺序排列。

### 切换 512 维 / 128 维模型

在 `src/pBox.h` 开头注释或打开对应宏，然后重新编译：

```c++
#define Num 512
//#define Num 128
```

## 关键实现细节

- **预处理**：图片缩放至 160x160 后做 prewhiten（逐图减均值、除调整后的标准差），BGR 转 RGB，按 CHW 通道优先排布写入网络输入。
- **残差缩放**：各 Inception-ResNet 块的捷径输出乘以缩放系数（0.17 / 0.1 / 0.2）后再与主路相加，与原论文一致。
- **输出归一化**：全连接输出做 L2 归一化后作为 embedding，因此欧氏距离与余弦相似度可互相换算：`distance² = 2 - 2·cosine`。
- **内存管理**：各子块计算完成后逐层释放中间 feature map 与权重缓冲，避免长序列推理时的内存泄漏。

## License

[GPL-2.0 License](LICENSE)
