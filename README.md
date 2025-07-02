# yolov5s 车辆行人检测系统
### (1) 下载和配置数据
1. 下载和转换数据  
使用 SODA10M 有标签的子数据集，可官网下载，也可百度云链接：[soda10m](https://pan.baidu.com/s/1KRJW_gcc_Zox6pOL4tWboA) ，提取码: cpdd

数据集包括 labeled_trainval.tar 和 labeled_test.tar 两个压缩包，数据集的标签文件是集中式的 json 文件，即所有图像的标签都在一个 json 文件中，json 文件有三个：instance_train.json、instance_val.json 和 instance_test.json，需要转换的是 instance_train.json、instance_val.json ，转换为 yolo 支持的格式，即 txt 文件，一个 txt 文件代表一幅图像的数据，txt 文件中的每一行代表一个标注框的数据：类别 + 归一化后的四个坐标点，共五个数据。转换方法见[coco_to_yolo](./coco_to_yolo/json_to_yolo.py)  
注意：原来的 json 文件中 6 个类别的 category_id 是 1 到 6 ，转换后是 0 到 5，即 ['Pedestrian':0, 'Cyclist':1, 'Car':2, 'Truck':3, 'Tram':4, 'Tricycle':5]，目的是为了方便 yolov5 模型训练，转换之后图像和标签数据按如下结构组织：
```
soda10m/
├── images/
│   ├── train/    # 训练图像（5000 张）
│   ├── val/      # 验证图像（5000 张）
│   └── test/     # 测试图像（10000 张）
└── labels/
    ├── train/    # 训练标签（5000 个）
    ├── val/      # 验证标签（5000 个）
    └── test/     # 测试标签（0 个，instance_test.json 中没有标注框的信息）
```
2. 创建数据集配置文件  
下载 yolov5 的 GitHub 仓库：`git clone https://github.com/ultralytics/yolov5.git`   
全部文件结构如下：
```
yolov5s/
├── soda10m/
│   ├── images/           # 全部图像（20000 张）
│   │       ├── train/    # 训练图像（5000 张）
│   │       ├── val/      # 验证图像（5000 张）
│   │       └── test/     # 测试图像（10000 张）
│   └── labels/           # 全部标签（10000 个）
│           ├── train/    # 训练标签（5000 个）
│           ├── val/      # 训练标签（5000 个）
│           └── test/     # 验证标签（0 个）
│
├── colo_to_yolo/   # 标签格式转换
│   ├── instance_test.json      # 测试集的标签文件，没有标注框信息
│   ├── instance_train.json     # 训练集的标签文件
│   ├── instance_val.json       # 验证集的标签文件
│   └── coco_to_yolo.py         # 转换 json 到 yolo  
│
└── yolov5/               # yolov5 仓库   
```
在 yolov5/data/ 下创建 soda10m.yaml ，内容如下：
```yaml
# soda10m.yaml 
# 路径根据实际修改
train: ../soda10m/images/train
val: ../soda10m/images/val
test: ../soda10m/images/test
nc: 6  # 类别数
names: ['Pedestrian', 'Cyclist', 'Car', 'Truck', 'Tram', 'Tricycle']
```
### (2) 模型训练配置
1. 环境准备  
```
# 终端执行下面命令创建新的环境 soda10m_yolov5s
conda create --name soda10m_yolov5s python>=3.9 

# 在终端打开 yolov5 文件夹，使用 activate soda10m_yolov5s 激活环境，
# 然后根据 yolov5 中的 requirements.txt 文件夹安装需要的包，执行下面命令
pip install -r requirements.txt

# 注意，这样安装的 torch 是 cpu 版本，不支持 gpu，所以 torch 需要另外安装才能使用 GPU 。
# 验证 torch 是否为 gpu 版本，执行：print(torch.cuda.is_available())，应返回 True
```
2. 启动训练  
使用 vscode，打开项目文件，环境选择之前创建的 soda10m_yolov5s 
在启动训练前修改一下 yolov5/train.py 中的 noautoanchor 参数，因为运行一次之后发现 原来的yolov5s.yaml 中的锚框尺寸和使用的数据集是兼容的，所以之后训练就不用再检查了，直接永久关闭，避免每次都在训练命令中指定关闭，修改如下：
```python
# 原代码
parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")

# 修改为
parser.add_argument("--noautoanchor", action="store_true",default=True, help="disable AutoAnchor")
```
注意：必须先进入 yolov5 目录，才能执行训练，因为上面配置文件中的路径都是相对 yolov5 的。  
执行下面命令启动训练。其中 yolov5s.pt 预训练权重要预先下载放入 yolov5 文件夹中。
```
python train.py \
  --img 640 \                  # 输入分辨率
  --batch 8 \                  # 批量大小（根据 GPU 显存调整）
  --epochs 100 \               # 训练轮次
  --data data/soda10m.yaml \   # 数据集配置
  --cfg models/yolov5s.yaml \  # 模型结构
  --weights yolov5s.pt \       # 预训练权重
  --name soda10m_yolov5s \     # 实验名称
  --rect \                     # 矩形训练（减少填充）
  --cos-lr \                   # 余弦学习率调度
  --label-smoothing 0.1        # 标签平滑防过拟合
  --workers 1                  # 进程数，默认是 8，

# 完全的命令如下，可自行添加或删除参数。
python train.py --img 640 --batch 8 --epochs 100 --data data/soda10m.yaml --cfg models/yolov5s.yaml --weights yolov5s.pt --name soda10m_yolov5s --rect --cos-lr --label-smoothing 0.1 --workers 1
```
3. 训练结果  
训练结果在 runs 文件夹中，包含权重、参数配置、训练数据等。  
训练结束后会保存 best.pt 和 last.pt 两个模型，运行下面命令执行验证脚本 val.py
```
python val.py --data data/soda10m.yaml --weights runs/train/soda10m_yolov5s/weights/best.pt
```
评价模型效果的核心指标：mAP@0.5（IoU=0.5 时的平均精度）、mAP@0.5:0.95（多 IoU 阈值平均）。由于测试集没有对应的标签，所以需要到 soda10m 数据集对应的官网上进行验证。  
### (3) 模型部署应用 
* 转为 ONNX(open neural network exchange，开放神经网络交换) 
```
python export.py --weights runs/train/soda10m_yolov5s/weights/best.pt --include onnx
```
* TensorRT 加速
```
python export.py --weights best.pt --include engine --device 0
```
* 推理示例  
```python
from yolov5 import detect
detect.run(
  weights='runs/train/soda10m_yolov5s/weights/best.pt',
  source='test_video.mp4',  # 图片/视频/摄像头
  conf_thres=0.25,          # 置信度阈值
  imgsz=640
)
```