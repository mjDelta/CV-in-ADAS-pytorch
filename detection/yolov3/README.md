
@[TOC](目录)
# 1 环境搭建
linux下
```
git clone https://github.com/ultralytics/yolov3.git 
```
在自己的对应的conda环境下，cd到对用的yolov3文件夹中，安装所需要的包
```
pip install -r requirements.txt 
```
要求
- python >= 3.6
- numpy
- torch >= 1.0.0
- opencv-python
- tqdm
# 2 数据集构造 
## voc格式数据的构造
windows环境下建立如图所示的空文件夹
- VOCdevkit2007
	- VOC2007
		-  Annotations       // 经过labellmg得到的xnl文件
		-  ImageSets
			- Main
		-  JPEGImages     // 先把你的jpg格式的图片放入这个文件夹
		-  labels
## 2.1 labellmg给数据画框得到xml格式文件
![labellmg工具的使用](https://img-blog.csdnimg.cn/20190527201435367.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
工具的使用: [目标检测标注工具labelImg使用方法](https://blog.csdn.net/xunan003/article/details/78720189).
xml保存路径选择到Annotations所在路径下，一个图片对应一个xml文件。

打开Annotations下和上图对应的xml文件
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019052720220442.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
object有两个就是刚才标注的汽车的框的位置 。

## 2.2 将样本打乱划分训练集等，生成txt文件
手动标注了100多张图片之后,╭(╯^╰)╮
VOC2007文件夹下运行该脚本 xml2txt. py.

```py
import os
import random

trainval_percent = 0.8
train_percent = 0.8
xmlfilepath = 'Annotations'
txtsavepath = 'ImageSets\Main'
total_xml = os.listdir(xmlfilepath)

num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open('ImageSets/Main/trainval.txt', 'w')
ftest = open('ImageSets/Main/test.txt', 'w')
ftrain = open('ImageSets/Main/train.txt', 'w')
fval = open('ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()
```
现在Main文件夹中有了4个txt文件，点进train.txt对应着经过划分后的训练集的所要用的图片
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527202807839.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
## 2.3 将对应图片名称的txt文件生成指向对应图片路径的txt文件
VOCdevkit2007文件夹==外==运行 voc_label.py
```py
"""
需要修改的地方：
1. sets中替换为自己的数据集
2. classes中替换为自己的类别
3. VOCdevkit2007文件夹外运行该脚本（对应生成的5个txt也在VOCdevkit2007文件夹外，
   并不重要，重要的是得到了txt文件）
4. 直接开始运行
"""
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

classes = ["car", "person"]


def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit2007/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit2007/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

wd = getcwd()

for year, image_set in sets:
    if not os.path.exists('VOCdevkit2007/VOC%s/labels/'%(year)):
        os.makedirs('VOCdevkit2007/VOC%s/labels/'%(year))
    image_ids = open('VOCdevkit2007/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('%s/VOCdevkit2007/VOC%s/JPEGImages/%s.jpg\n'%(wd, year, image_id))
        convert_annotation(year, image_id)
    list_file.close()

os.system("cat 2007_train.txt 2007_val.txt  > train.txt")
os.system("cat 2007_train.txt 2007_val.txt 2007_test.txt  > train.all.txt")
```
打开对应的2007_train.txt
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019052720411519.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
就是将原来的train.txt加了路径前缀和文件类型后缀嘛~ **这个文件之后是要修改的** ，**路径要改成linux下对应**的目录，因为我的linux没有图形界面，所以我是先windows制作好了数据集，更改路径，再传到了linux对应的路径中。
## 2.4 将xml格式的标注转化成txt形式的标注
voc_label.py后labels文件夹下也多了txt文件 （一个jpg对应一个xml标注和一个txt标注）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527205720520.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
打开一个txt看一下 第一个数字0即类别car（100多张图片我只标注了0car 1person两类），剩下的四个参数即标准化后框边界的位置（可以从程序中看出）
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527210007512.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)

VOC格式数据集构造完毕
文件夹结构
- VOCdevkit2007
	- VOC2007
		-  Annotations       // 经过labellmg得到的xnl文件
			- 00005.xml 
		-  ImageSets
			- Main
				-  train.txt
		-  JPEGImages     // 先把你的jpg格式的图片放入这个文件夹
			- <font color = blue size = 4>00005.jpg...
		-  labels
			- <font color = red size = 4>00005.txt...
-  <font color = purple size = 4>2007_train.txt </font>  //图片对应的路径 
# 3 制作对应的coco数据集格式
linux下
- yolov3
	- cfg
	- coco
	- data
	- weights
	
## 3.1 coco下创建image和labels文件夹，分别放所有图片和标准化后的所有labels
## 3.2 data下添加/更改coco.data,coco.names,2007_train.txt,2007_test.txt
在linux下进入yolov3文件夹，xftp下的linux的可视化
images文件夹下图片的路径复制到2007_test.txt中
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527213433199.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
修改2007_train.txt和2007_train.txt文件下对应的路径，修改完毕放入data文件夹下
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190527213539573.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
data文件夹下创建coco.names文件，注意第一行对应 0类汽车，依次类推
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019052721444459.png)
data文件夹下更改coco.data文件

```
classes=2  # 自己的类
train = ./data/2007_train.txt
valid = ./data/2007_test.txt   //在这里直接把验证集的路径写到了测试集里
names=data/coco.names
backup=backup/
```
当前的文件格式
- yolov3
	- cfg
	- coco
		- image
			- <font color = blue size = 3>00005.jpg... 
		- labels
			- <font color = red size = 3>00005.txt...
	- data
		- coco.data
		- coco.names 
		-  <font color = purple size = 3>2007_train.txt  </font> 
		-  <font color = purple size = 3>2007_test.txt </font>  
	- weights

# 4调整网络结构与配置
打开cfg文件夹下的<font color = red size = 3>yolov3.cfg</font>  
[net]部分，主要更改subdivision和batch部分
 [其他参数的意义](https://pprp.github.io/2018/09/20/tricks.html).
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190528131143376.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
[yolo]和[yolo]层上的[convolutional]
总共有3个yolo层需要更改
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190528133240956.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)

# 5 实战
## 5.1 训练
更改train.py中的设置，14行处加入，设置成单路GPU，多路会出现**address in use** 错误，未能解决
```py
os.environ['MASTER_PORT'] = '9901'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
```
命令行中，载入自己的数据和网络结构
```
python train.py --data data/coco.data --cfg cfg/yolov3.cfg
```
也可以在文件中将默认参数改为对应的路径， test. py  和 detect.py同理。
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190528133946832.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
第一次训练会加载weights文件下的预训练参数，之后的输出如下，每次迭代的结果的评价指标也都与显示，对测试集36张图片的mPA计算
```
Namespace(accumulate=1, backend='nccl', batch_size=16, cfg='cfg/yolov3.cfg', data_cfg='data/coco.data', dist_url='tcp://127.0.0.1:9999', epochs=273, evolve=False, img_size=416, multi_scale=False, nosave=False, notest=False, num_workers=4, rank=0, resume=False, transfer=False, var=0, world_size=1)
Using CUDA device0 _CudaDeviceProperties(name='GeForce GTX 1080 Ti', total_memory=11178MB)

Reading images: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 115/115 [00:01<00:00, 87.79it/s]
Model Summary: 222 layers, 6.15291e+07 parameters, 6.15291e+07 gradients

   Epoch       Batch        xy        wh      conf       cls     total  nTargets      time
QXcbConnection: Failed to initialize XRandr
   0/272         0/7      0.85      1.69       151      1.28       154        78      84.4
   0/272         1/7     0.804      1.73       151       1.3       154        73     0.314
   0/272         2/7     0.784      1.74       151      1.31       154        71     0.312
   0/272         3/7     0.777       1.7       150      1.31       153        90      0.31
   0/272         4/7     0.778       1.6       145       1.3       148        81     0.311
   0/272         5/7     0.771      1.51       138       1.3       141        80     0.313
   0/272         6/7     0.765      1.41       130      1.28       134        80      0.31
   0/272         7/7     0.684      1.26       115      1.15       118        11       1.9
Reading images: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 36/36 [00:00<00:00, 80.61it/s]
               Class    Images   Targets         P         R       mAP        F1
Computing mAP: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  3.35s/it]
                 all        36       114         0         0         0         0

                 car        36        73         0         0         0         0
              person        36        41         0         0         0         0

```
迭代完成后的结果在yolov3文件夹下的 results.txt 中，在juputer中运行一下代码可视化了迭代的参数变化

```py
import glob
import random

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
def plot_results(start=0, stop=0):  # from utils.utils import *; plot_results()
    # Plot training results files 'results*.txt'
    # import os; os.system('wget https://storage.googleapis.com/ultralytics/yolov3/results_v3.txt')

    fig, ax = plt.subplots(2, 5, figsize=(14, 7))
    ax = ax.ravel()
    s = ['X + Y', 'Width + Height', 'Confidence', 'Classification', 'Train Loss', 'Precision', 'Recall', 'mAP', 'F1',
         'Test Loss']
    for f in sorted(glob.glob('results.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 5, 6, 9, 10, 11, 12, 13]).T
        n = results.shape[1]  # number of rows
        x = range(start, min(stop, n) if stop else n)
        for i in range(10):
            ax[i].plot(x, results[i, x], marker='.', label=f.replace('.txt', ''))
            ax[i].set_title(s[i])
    fig.tight_layout()
    ax[4].legend()
    fig.savefig('results.png', dpi=300)
plot_results()
```
类似这样，100多张训练的结果一般般，数据多了精度自然会上去
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190528140918416.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)

## 5.2 检测
data文件夹下的samples文件夹下放入要检测的图片。
命令行中，输入
```
python3 detect.py --weights weights/latest.pt
```

```
image 1/8 data/samples/00000.jpg: 160x416 2 cars, Done. (0.146s)
image 2/8 data/samples/00005.jpg: 160x416 2 cars, Done. (0.017s)
image 3/8 data/samples/01050.jpg: 160x416 2 cars, Done. (0.017s)
image 4/8 data/samples/01200.jpg: Done. (0.013s)
image 5/8 data/samples/02220.jpg: Done. (0.013s)
image 6/8 data/samples/02430.jpg: 160x416 2 cars, Done. (0.017s)
image 7/8 data/samples/bus.jpg: 416x320 1 persons, Done. (0.018s)
image 8/8 data/samples/zidane.jpg: Done. (0.016s)
Results saved to /home/***/yolov3/output
```
很小的训练集达到了还可以的效果，可以多制作图片来提升精度
![在这里插入图片描述](https://img-blog.csdnimg.cn/2019052814200963.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190528141816335.jpg?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3FxXzQxNTY1OTIw,size_16,color_FFFFFF,t_70)
# 6 参考
 [yolov3官网](https://pjreddie.com/darknet/yolo/).
 [目标检测：YOLOv3: 训练自己的数据](https://blog.csdn.net/lilai619/article/details/79695109).
 [github:Train Custom Data](https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data).
 [pytorch版yolov3训练自己数据集](https://www.cnblogs.com/pprp/p/10863496.html)
