#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-21 12:15:26
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from models import *
import torch 
from data_utils import *
import numpy as np
import os
import cv2

USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")

lane_imgs_dir="E:/cv-adas/driver_161_90frame/driver_161_90frame"
lane_labels_dir="E:/cv-adas/laneseg_label_w16/laneseg_label_w16/driver_161_90frame"
out_dir="E:/cv-adas/out-driver_161_90frame-unet2d/"
model_path=os.path.join(out_dir,"epoch_49.tar")

size_h=16*8
size_w=size_h*3
batch_size=1
train_rate=0.8

unet=UNet2D()
unet=unet.to(device)
unet.train(mode=False)

model_sd=torch.load(model_path)
unet.load_state_dict(model_sd["unet"])

video_paths,label_paths=get_video_label_paths(lane_imgs_dir,lane_labels_dir)
rng=np.random.RandomState(0)
all_idxs=np.arange(len(video_paths))
rng.shuffle(all_idxs)
train_idxs=all_idxs[:int(len(video_paths)*train_rate)]
val_idxs=all_idxs[int(len(video_paths)*train_rate)]
test_idxs=all_idxs[int(len(video_paths)*train_rate)+1:]

test_video_paths=video_paths[test_idxs]
test_laebl_paths=label_paths[test_idxs]

for test_video_path,test_label_path in zip(test_video_paths,test_laebl_paths):
	batch_splits=len(test_video_path)//batch_size
	video_name=test_video_path[0].replace("\\","/").split("/")[-2]
	video_output_path=os.path.join(out_dir,video_name)
	video_writer=cv2.VideoWriter(video_output_path,cv2.VideoWriter_fourcc(*'MPEG'),5,(size_w,size_h))
	for batch_split in range(batch_splits):
		start=batch_split*batch_size
		end=(batch_split+1)*batch_size
		batch_imgs,batch_labels=read_video_mask(test_video_path[start:end],test_label_path[start:end],size_h,size_w)
		batch_imgs_tensor=torch.FloatTensor(batch_imgs).to(device)

		batch_preds=unet(batch_imgs_tensor)

		batch_preds_arr=batch_preds.detach().cpu().numpy()

		for img,label,pred in zip(batch_imgs,batch_labels,batch_preds_arr):
			img=np.transpose(img,(1,2,0))
			pred=np.squeeze(pred)
			pred_max=pred.max()
			pred_min=pred.min()
			pred=(pred-pred_min)/(pred_max-pred_min)
			pred=pred>0.5

			# img_label=get_mask_on_image(img,label)
			img_pred=get_mask_on_image(img,pred)
			# show_img_label_pred(img,img_label,img_pred)
			video_writer.write(np.uint8(img_pred*255))
	video_writer.release()
	print("Video {} segmentation completed!".format(video_name))
