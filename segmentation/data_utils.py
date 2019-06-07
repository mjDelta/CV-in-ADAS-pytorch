#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-19 18:34:32
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import os
from skimage import io
from matplotlib import pyplot as plt
import numpy as np
from skimage.transform import resize
from scipy import linalg as sla
def compute_iou_cpu(preds,trues):
	iou=[]
	for pred,true in zip(preds,trues):
		intersection=np.where(np.multiply(pred,true),1,0).sum()
		union=np.where(pred+true!=0,1,0).sum()
		iou.append(intersection/(union+1e-7))
	return np.array(iou).mean()
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
def compute_beta(size_w,size_h,mask_list):
	betas=[]

	for mask in mask_list:
		x_map,y_map=generate_xy_coordinates_by_labels(size_w,size_h,mask)

		if x_map.shape[0]==0:
			betas.append(-1)
			continue
		XtX,XtY=x_map.T.dot(x_map),x_map.T.dot(y_map)
		beta=sla.solve(XtX,XtY,sym_pos=True,check_finite=False)[0]
		betas.append(beta)
	return betas
def read_video_mask_4lanes(img_paths,label_paths,size_h,size_w):
	imgs=[]
	masks=[]
	betas=[]
	for img_path,label_path in zip(img_paths,label_paths):
		img,label=read_img_label(img_path,label_path)
		tmp_mask=[]
		mask1=label==1;mask1=mask1.astype(np.float32);mask1=resize(mask1,(size_h,size_w));mask1=mask1!=0;mask1=mask1.astype(np.float32);tmp_mask.append(mask1)
		mask2=label==2;mask2=mask2.astype(np.float32);mask2=resize(mask2,(size_h,size_w));mask2=mask2!=0;mask2=mask2.astype(np.float32);tmp_mask.append(mask2)
		mask3=label==3;mask3=mask3.astype(np.float32);mask3=resize(mask3,(size_h,size_w));mask3=mask3!=0;mask3=mask3.astype(np.float32);tmp_mask.append(mask3)
		mask4=label==4;mask4=mask4.astype(np.float32);mask4=resize(mask4,(size_h,size_w));mask4=mask4!=0;mask4=mask4.astype(np.float32);tmp_mask.append(mask4)
		masks.append(tmp_mask)
		tmp_betas=compute_beta(size_w,size_h,tmp_mask)
		betas.append(tmp_betas)
		img=resize(img,(size_h,size_w))
		img=np.transpose(img,(2,0,1))
		imgs.append(img)
	return np.array(imgs),np.array(masks),np.array(betas)
def generate_xy_coordinates_by_labels(size_w,size_h,label):
	x=np.arange(0,size_w)
	y=np.arange(0,size_h)
	x_coordinates,y_coordinates=np.meshgrid(x,y)
	x_map=np.multiply(x_coordinates,label)
	y_map=np.multiply(y_coordinates,label)
	nonzeros=np.nonzero(label)
	return x_map[nonzeros],y_map[nonzeros]
def generate_xy_coordinates(size_w,size_h):
	x=np.arange(0,size_w)
	y=np.arange(0,size_h)
	x_coordinates,y_coordinates=np.meshgrid(x,y)
	return x_coordinates,y_coordinates
if __name__=="__main__":
	# lane_imgs_dir="E:/cv-adas/driver_161_90frame/driver_161_90frame"
	# lane_labels_dir="E:/cv-adas/laneseg_label_w16/laneseg_label_w16/driver_161_90frame"
	# size_h=16*8
	# size_w=size_h*3
	# img_paths,label_paths=get_video_label_paths(lane_imgs_dir,lane_labels_dir)
	# for i in [20,21,22]:
	# 	full_path_img=img_paths[10][i]
	# 	full_path_label=label_paths[10][i]
	# 	imgs,labels,betas=read_video_mask_4lanes([full_path_img],[full_path_label],size_h,size_w)
	# 	print(betas)
	# 	img,label1,label2,label3,label4=np.transpose(imgs[0],(1,2,0)),labels[0,0],labels[0,1],labels[0,2],labels[0,3]
	# 	fig=plt.figure()
	# 	fig.add_subplot(511)
	# 	plt.imshow(img)
	# 	fig.add_subplot(512)
	# 	plt.imshow(label1)
	# 	fig.add_subplot(513)
	# 	plt.imshow(label2)
	# 	fig.add_subplot(514)
	# 	plt.imshow(label3)
	# 	fig.add_subplot(515)
	# 	plt.imshow(label4)
	# 	plt.show()
	pred=np.array([[0,1,0],[1,1,1],[0,1,0]])
	true=np.array([[1,0,0],[0,1,0],[0,0,1]])
	print(compute_iou_cpu(pred,true))
