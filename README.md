# CV-in-ADAS
  ADAS system features *night vision assistance, anti-collision alert, collision resolution, lane departure warning, lane keeping, lane change assist*, corner obstacle detection, driver status monitoring, driver reminder, parking assistance, traffic sign recognition, and high beam Auxiliary and so on.</br>
  From the system function, the ADAS system has a significant number of tasks related to computer vision (CV) which are as follows:</br>
1. Detection part: vehicle detection, pedestrian detection, non-motor vehicle detection, traffic sign recognition;</br>
2. Segmentation: lane line detection, determination of the travelable area.</br>

  In order to realize the above functions, it is necessary to separately establish a model for each function. However, parameters for these models are so many that the performance of the vehicle **real-time** system may be influenced.</br>
## Semantic Segmentation
| Task | Purpose |
| :--: | :--: |
| Lane segmentation | Lane departure warning |
## Detection
| Task | Purpose |
| :--: | :--: |
| Vihicle detection | Vehicle approach warning |
| Pedestrain detection | Pedestrian approach warning |
## Datasets
Lane Segmentation: https://xingangpan.github.io/projects/CULane.html

## To do list
- [ ] Segmentation: FCN series
  - [x] 2D U-Net Seg
  - [ ] 3D U-Net Seg
- [ ] Detection: YOLO series
- [ ] Combining the above
