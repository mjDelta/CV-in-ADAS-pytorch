#!//usr//bin//env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-19 17:35:51
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

import torch
import torch.nn as nn

class UNet2D(nn.Module):
	def __init__(self):
		super(UNet2D,self).__init__()

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
			nn.LeakyReLU(),
			nn.Conv2d(32,1,1,padding=0),
			nn.Sigmoid())
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
		y=self.final_block(tmp_de1)
		return y

class UNet2D_4lanes(nn.Module):
	def __init__(self):
		super(UNet2D_4lanes,self).__init__()

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
		y=self.final_block(tmp_de1)
		return y


