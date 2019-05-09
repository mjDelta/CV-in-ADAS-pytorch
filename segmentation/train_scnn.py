#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-04 12:04:35
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from models_scnn import *
import torch 
from torch import optim
from data_utils import *
import numpy as np
import torch.nn as nn
import math
from visdom import Visdom
from tqdm import tqdm
import pandas as pd

def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

USE_CUDA=torch.cuda.is_available()
device=torch.device("cuda" if USE_CUDA else "cpu")

lane_imgs_dir="E:/cv-adas/driver_161_90frame/driver_161_90frame"
lane_labels_dir="E:/cv-adas/laneseg_label_w16/laneseg_label_w16/driver_161_90frame"
out_dir="E:/cv-adas/out-driver_161_90frame-scnn2d/"
mkdirs(out_dir)
epochs=50
size_h=16*8
size_w=size_h*3
batch_size=10
train_rate=0.8

vis=Visdom(env="seg-scnn")
scnn=SCNN2D()
scnn=scnn.to(device)
scnn.train(mode=True)
optimizer=optim.SGD(scnn.parameters(),lr=1e-4,momentum=0.9,weight_decay=0.005)
criterion=nn.BCELoss()

video_paths,label_paths=get_video_label_paths(lane_imgs_dir,lane_labels_dir)
rng=np.random.RandomState(0)
all_idxs=np.arange(len(video_paths))
rng.shuffle(all_idxs)
train_idxs=all_idxs[:int(len(video_paths)*train_rate)]
val_idxs=all_idxs[int(len(video_paths)*train_rate)]
test_idxs=all_idxs[int(len(video_paths)*train_rate)+1:]

val_video_path=video_paths[val_idxs]
val_label_path=label_paths[val_idxs]

iteration=0
iter_losses=[]
train_losses=[]
val_losses=[]
for epoch in range(epochs):
	rng.shuffle(train_idxs)
	train_video_paths=video_paths[train_idxs]
	train_label_paths=label_paths[train_idxs]
	##train
	scnn.train(mode=True)
	epoch_train_loss=0.
	for video_path,label_path in zip(train_video_paths,train_label_paths):
		if len(video_path)==0:continue
		batch_splits=len(video_path)//batch_size
		tmp=0.
		for batch_split in range(batch_splits):
			optimizer.zero_grad()
			start=batch_split*batch_size
			end=(batch_split+1)*batch_size

			batch_imgs,batch_labels=read_video_mask(video_path[start:end],label_path[start:end],size_h,size_w)

			batch_imgs=torch.FloatTensor(batch_imgs).to(device)
			batch_labels=torch.FloatTensor(batch_labels).to(device)
			scnn_output=scnn(batch_imgs)
			scnn_output=torch.squeeze(scnn_output)
			scnn_output_flat=scnn_output.view(-1)
			batch_labels_flat=batch_labels.view(-1)
			batch_loss=criterion(scnn_output_flat,batch_labels_flat)
			vis.line(X=torch.LongTensor([iteration]),Y=torch.FloatTensor([batch_loss.item()]),win="iteration loss",update="append",opts={"title":"train loss(iteration)"})
			iter_losses.append(batch_loss.item())
			tmp+=batch_loss.item()
			iteration+=1
			batch_loss.backward()
			optimizer.step()

		tmp/=batch_splits
		epoch_train_loss+=tmp
		
	epoch_train_loss/=len(train_idxs)
	torch.cuda.empty_cache()
	##val
	scnn.train(mode=False)
	epoch_val_loss=0.
	val_batch_splits=len(val_video_path)//batch_size
	for batch_split in range(val_batch_splits):
		start=batch_split*batch_size
		end=(batch_split+1)*batch_size

		batch_imgs,batch_labels=read_video_mask(val_video_path[start:end],val_label_path[start:end],size_h,size_w)

		batch_imgs=torch.FloatTensor(batch_imgs).to(device)
		batch_labels=torch.FloatTensor(batch_labels).to(device)
		scnn_output=scnn(batch_imgs)
		scnn_output=torch.squeeze(scnn_output)
		scnn_output_flat=scnn_output.view(-1)
		batch_labels_flat=batch_labels.view(-1)
		batch_loss=criterion(scnn_output_flat,batch_labels_flat)
		epoch_val_loss+=batch_loss.item()	
		vis.image(batch_imgs[0],win="org img",opts={"title":"org img"})
		vis.image(scnn_output[0],win="labeled img",opts={"title":"pred img"})
		vis.image(batch_labels[0],win="label",opts={"titel":"label"})
	epoch_val_loss/=val_batch_splits	
	vis.line(X=torch.LongTensor([epoch]),Y=torch.FloatTensor([epoch_train_loss]),win="epoch train loss",update="append",opts={"title":"train loss(epoch)"})
	vis.line(X=torch.LongTensor([epoch]),Y=torch.FloatTensor([epoch_val_loss]),win="epoch val loss",update="append",opts={"title":"val loss(epoch)"})
	train_losses.append(epoch_train_loss)
	val_losses.append(epoch_val_loss)
	print("Epoch {}: trian loss {}\tval loss {}".format(epoch,epoch_train_loss,epoch_val_loss))	
	torch.save({
		"scnn":scnn.state_dict(),
		"optimizer":optimizer.state_dict()
		},os.path.join(out_dir,"epoch_{}.tar".format(epoch)))

df1=pd.DataFrame()
df1["iteration"]=np.arange(len(iter_losses))
df1["loss"]=iter_losses
df1.to_csv(os.path.join(out_dir,"iter_loss.csv"),index=False)

df1=pd.DataFrame()
df1["epochs"]=np.arange(len(train_losses))
df1["loss"]=train_losses
df1.to_csv(os.path.join(out_dir,"epoch_train_loss.csv"),index=False)

df1=pd.DataFrame()
df1["epochs"]=np.arange(len(val_losses))
df1["loss"]=val_losses
df1.to_csv(os.path.join(out_dir,"epoch_val_loss.csv"),index=False)