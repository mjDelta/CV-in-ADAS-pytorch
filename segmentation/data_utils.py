#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-19 18:34:32
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import os
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)
def get_mask_on_image(image,label):
	mask=label!=0
	masked_img=image.copy()
	masked_img[:,:,0][mask]=0
	masked_img[:,:,1][mask]=1
	masked_img[:,:,2][mask]=0
	return masked_img
def show_img_label_pred(img,label,pred):
	fig=plt.figure(figsize=(10,15))
	fig.add_subplot(311)
	plt.title("IMG")
	plt.imshow(img)
	fig.add_subplot(312)
	plt.title("LABEL")
	plt.imshow(label)
	fig.add_subplot(313)
	plt.title("PRED")
	plt.imshow(pred)
	plt.show()	
def get_video_label_paths(lane_imgs_dir,lane_labels_dir):
	video_paths=[]
	label_paths=[]
	for d in os.listdir(lane_imgs_dir):
		full_dir_img=os.path.join(lane_imgs_dir,d)
		tmp_imgs=[]
		tmp_labels=[]
		for f in os.listdir(full_dir_img):
			if os.path.splitext(f)[1]!=".jpg":
				continue
			full_path_img=os.path.join(full_dir_img,f)
			full_path_label=os.path.join(lane_labels_dir,d,os.path.splitext(f)[0]+".png")	
			tmp_imgs.append(full_path_img)
			tmp_labels.append(full_path_label)
		video_paths.append(tmp_imgs)
		label_paths.append(tmp_labels)
	return np.array(video_paths),np.array(label_paths)
def read_img_label(img_path,label_path):
	img=io.imread(img_path)
	label=io.imread(label_path)	
	return img,label
def read_video_mask(img_paths,label_paths,size_h,size_w):
	imgs=[]
	masks=[]
	for img_path,label_path in zip(img_paths,label_paths):
		img,label=read_img_label(img_path,label_path)
		img=resize(img,(size_h,size_w))
		label=resize(label,(size_h,size_w))
		mask=label!=0
		mask=mask.astype(np.float32)
		# print(img.max())
		# img=img/255.
		img=np.transpose(img,(2,0,1))
		imgs.append(img)
		masks.append(mask)
	return np.array(imgs),np.array(masks)

if __name__=="__main__":
	lane_imgs_dir="E:/cv-adas/driver_161_90frame/driver_161_90frame"
	lane_labels_dir="E:/cv-adas/laneseg_label_w16/laneseg_label_w16/driver_161_90frame"
	size_h=16*8
	size_w=size_h*3
	img_paths,label_paths=get_video_label_paths(lane_imgs_dir,lane_labels_dir)
	full_path_img=img_paths[10][30]
	full_path_label=label_paths[10][30]
	imgs,labels=read_video_mask([full_path_img],[full_path_label],size_h,size_w)
	img,label=np.transpose(imgs[0],(1,2,0)),labels[0]
	labeled_img=get_mask_on_image(img,label)
	fig=plt.figure(figsize=(10,15))
	fig.add_subplot(311)
	plt.imshow(img)
	fig.add_subplot(312)
	plt.imshow(label)
	fig.add_subplot(313)
	plt.imshow(labeled_img)
	plt.show()
	print(img.max())
	print(label.max())
	print(img.shape)
	print(label.shape)
