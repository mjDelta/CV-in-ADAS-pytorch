# CV-in-ADAS
ADAS is short for Advanced Driver Assistance System. And it has the following features *night vision assistance, anti-collision alert, collision resolution, lane departure warning, lane keeping, lane change assist*, corner obstacle detection, driver status monitoring, driver reminder, parking assistance, traffic sign recognition, and high beam Auxiliary and so on.</br>

From the system function, the ADAS system has a significant number of tasks related to computer vision (CV) which are as follows:
1. Detection: vehicle detection, pedestrian detection, non-motor vehicle detection, traffic sign recognition;
2. Segmentation: lane detection.

In order to realize the above functions, it is necessary to separately establish a model for each function. However, parameters for these models are so many that the performance of the vehicle **real-time** system may be influenced.</br>
## 0. Environment
Python version: 3.5</br>
Deep learning framework: pytorch</br></br>
package list:
1. scikit-image (0.12.3)
2. numpy (1.16.2)
3. torch (1.0.1)
4. torchvision (0.2.2.post3)
5. visdom (0.1.8.8)
6. matplotlib (1.5.3)
7. pandas (0.18.1)
8. opencv-python (3.4.3.18)
9. tqdm (4.26.0)
## 1. Semantic Segmentation
| Task | Purpose |
| :--: | :--: |
| Lane segmentation | Lane departure warning |
## 2. Detection
| Task | Purpose |
| :--: | :--: |
| Vehicle detection | Vehicle approach warning |
| Pedestrain detection | Pedestrian approach warning |
## 3. Datasets
Lane Segmentation: https://xingangpan.github.io/projects/CULane.html
## 4. Performance
### 4.1 Semantic segmentation
| Model | BCE Loss | IOU | ACC | FPS|
| :--: | :--: | :--: | :--: | :--: |
| Unet-2D | 0.0620 | \ | \ | \ |
| VPGNet | \ | \ | \ | \ |
| S-CNN | 0.0371 | \ | \ | \ |
| Least Square Fitting | \ | \ | \ | \ |
### 4.2 Result display
![unet2d-seg](https://github.com/mjDelta/CV-in-ADAS/blob/master/img/seg_unet2d.png)
![scnn2d-seg](https://github.com/mjDelta/CV-in-ADAS/blob/master/img/seg_scnn.png)

![unet2d-loss](https://github.com/mjDelta/CV-in-ADAS/blob/master/img/loss_UNET.png)
![scnn2d-loss](https://github.com/mjDelta/CV-in-ADAS/blob/master/img/loss_SCNN.png)

### 4.2 Detection
## 5. To do list
- [ ] Segmentation
  - [x] Traditional segmentation method
  - [x] 2D U-Net Seg
  - [ ] VPGNet
  - [x] S-CNN
  - [x] Least Square Fitting
- [ ] Detection
  - [ ] YOLO
  - [ ] Faster-RCNN
  - [ ] Stereo-RCNN
- [ ] Combining the above
## 6. Reference paper
### 6.1 Segmentation part
1. [Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation[C]//International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015: 234-241.](https://arxiv.org/abs/1505.04597)
2. [Pan X, Shi J, Luo P, et al. Spatial as deep: Spatial cnn for traffic scene understanding[C]//Thirty-Second AAAI Conference on Artificial Intelligence. 2018.](https://arxiv.org/abs/1712.06080)
3. [Lee S, Kim J, Shin Yoon J, et al. Vpgnet: Vanishing point guided network for lane and road marking detection and recognition[C]//Proceedings of the IEEE International Conference on Computer Vision. 2017: 1947-1955.](https://arxiv.org/abs/1710.06288)
4. [De Brabandere B , Van Gansbeke W , Neven D , et al. End-to-end Lane Detection through Differentiable Least-Squares Fitting[J]. 2019.](https://arxiv.org/abs/1902.00293v1)
### 6.2 Detection part
1. [Ren S, He K, Girshick R, et al. Faster r-cnn: Towards real-time object detection with region proposal networks[C]//Advances in neural information processing systems. 2015: 91-99.](https://arxiv.org/pdf/1506.01497.pdf)
2. [Li P, Chen X, Shen S. Stereo R-CNN based 3D Object Detection for Autonomous Driving[J]. arXiv preprint arXiv:1902.09738, 2019.](https://arxiv.org/pdf/1902.09738.pdf)
3. [Redmon J, Farhadi A. Yolov3: An incremental improvement[J]. arXiv preprint arXiv:1804.02767, 2018.](https://pjreddie.com/media/files/papers/YOLOv3.pdf)
