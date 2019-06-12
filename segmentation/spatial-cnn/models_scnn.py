#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-05-04 11:42:12
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import torch
import torch.nn as nn
import torch.nn.functional as F
class SCNN2D(nn.Module):
	def __init__(self):
		super(SCNN2D,self).__init__()

		self.encoder_block1=nn.Sequential(
			nn.Conv2d(3,32,3,padding=(3-1)//2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.Conv2d(32,32,3,padding=(3-1)//2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU())
		self.encoder_block1_pool=nn.MaxPool2d(kernel_size=2,stride=2)
		self.encoder_block2=nn.Sequential(
			nn.Conv2d(32,64,3,padding=(3-1)//2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Conv2d(64,64,3,padding=(3-1)//2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU())
		self.encoder_block2_pool=nn.MaxPool2d(kernel_size=2,stride=2)
		self.encoder_block3=nn.Sequential(
			nn.Conv2d(64,128,3,padding=(3-1)//2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128,128,3,padding=(3-1)//2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU())
		self.encoder_block3_pool=nn.MaxPool2d(kernel_size=2,stride=2)
		self.encoder_block4=nn.Sequential(
			nn.Conv2d(128,256,3,padding=(3-1)//2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(),
			nn.Conv2d(256,256,3,padding=(3-1)//2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU())
		self.encoder_block4_pool=nn.MaxPool2d(kernel_size=2,stride=2)

		self.decoder_block4=nn.Sequential(
			nn.Conv2d(256,256,3,padding=(3-1)//2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(),
			nn.Conv2d(256,256,3,padding=(3-1)//2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True))
		self.decoder_block3=nn.Sequential(
			nn.Conv2d(512,256,3,padding=(3-1)//2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(),
			nn.Conv2d(256,256,3,padding=(3-1)//2),
			nn.BatchNorm2d(256),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True))
		self.decoder_block2=nn.Sequential(
			nn.Conv2d(384,128,3,padding=(3-1)//2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Conv2d(128,128,3,padding=(3-1)//2),
			nn.BatchNorm2d(128),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True))
		self.decoder_block1=nn.Sequential(
			nn.Conv2d(192,64,3,padding=(3-1)//2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Conv2d(64,64,3,padding=(3-1)//2),
			nn.BatchNorm2d(64),
			nn.LeakyReLU(),
			nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True))
		self.final_block=nn.Sequential(
			nn.Conv2d(96,32,3,padding=(3-1)//2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU(),
			nn.Conv2d(32,32,3,padding=(3-1)//2),
			nn.BatchNorm2d(32),
			nn.LeakyReLU())

		self.conv_d=nn.Conv2d(32,32,(1,5),padding=(0,2),bias=False)
		self.conv_u=nn.Conv2d(32,32,(1,5),padding=(0,2),bias=False)
		self.conv_r=nn.Conv2d(32,32,(5,1),padding=(2,0),bias=False)
		self.conv_l=nn.Conv2d(32,32,(5,1),padding=(2,0),bias=False)

		self.dropout=nn.Dropout(0.1)
		self.conv=nn.Conv2d(32,1,1)
		self.activation=nn.Sigmoid()

	def forward(self,x):
		en1=self.encoder_block1(x)
		pool1=self.encoder_block1_pool(en1)
		en2=self.encoder_block2(pool1)
		pool2=self.encoder_block2_pool(en2)
		en3=self.encoder_block3(pool2)
		pool3=self.encoder_block3_pool(en3)
		en4=self.encoder_block4(pool3)
		pool4=self.encoder_block4_pool(en4)

		de4=self.decoder_block4(pool4)#256
		tmp_de4=torch.cat([de4,en4],dim=1)

		de3=self.decoder_block3(tmp_de4)
		tmp_de3=torch.cat([de3,en3],dim=1)

		de2=self.decoder_block2(tmp_de3)
		tmp_de2=torch.cat([de2,en2],dim=1)		

		de1=self.decoder_block1(tmp_de2)
		tmp_de1=torch.cat([de1,en1],dim=1)

		top_hidden=self.final_block(tmp_de1)

		for i in range(1,top_hidden.shape[2]):
			top_hidden[...,i:i+1,:].add_(F.relu(self.conv_d(top_hidden[...,i-1:i,:])))
		for i in range(top_hidden.shape[2]-2,0,-1):
			top_hidden[...,i:i+1,:].add_(F.relu(self.conv_u(top_hidden[...,i+1:i+2,:])))
		for i in range(1,top_hidden.shape[3]):
			top_hidden[...,i:i+1].add_(F.relu(self.conv_r(top_hidden[...,i-1:i])))
		for i in range(top_hidden.shape[3]-2,0,-1):
			top_hidden[...,i:i+1].add_(F.relu(self.conv_l(top_hidden[...,i+1:i+2])))

		y=self.dropout(top_hidden)
		y=self.conv(y)
		y=self.activation(y)

		return y