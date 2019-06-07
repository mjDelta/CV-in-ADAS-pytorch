#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-02 21:22:24
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)
import torch
from torch import nn
import numpy as np
from scipy import linalg as sla
class UNet2D_4Lanes_LSF(nn.Module):
	def __init__(self,feature_extractor,device):
		super(UNet2D_4Lanes_LSF,self).__init__()
		self.feature_extractor=feature_extractor
		self.device=device
	def forward(self,x,masks):
		## masks: B,C,H,W
		weights=self.feature_extractor(x)
		B,C,H,W=weights.shape
		betas=[]
		for b in range(B):
			tmp_betas=[]
			for c in range(C):
				xv,yv=self.generate_coordinates(W,H,weights[b,c],masks[b,c])
				if xv.shape[0]==0:
					tmp_betas.append(-1)
					continue
				beta=self.compute_beta_cuda(xv,yv)
				tmp_betas.append(beta)
			betas.append(tmp_betas)
		betas=torch.FloatTensor(betas).to(self.device)

		return weights,betas
	def generate_coordinates(self,size_w,size_h,w,label):
		xv,yv=torch.meshgrid([torch.arange(0,size_h),torch.arange(0,size_w)])
		xv=xv.type(torch.FloatTensor);yv=yv.type(torch.FloatTensor)
		xv=xv.to(self.device);yv=yv.to(self.device)

		xv,yv=torch.mul(xv,w),torch.mul(yv,w)
		xv,yv=torch.mul(xv,label),torch.mul(yv,label)
		nonzeros=torch.nonzero(xv)
		return xv[nonzeros[:,0],nonzeros[:,1]].unsqueeze(1),yv[nonzeros[:,0],nonzeros[:,1]].unsqueeze(1)

	def compute_beta_cuda(self,X,Y):
		XtX,XtY=X.permute(1,0).mm(X),X.permute(1,0).mm(Y)
		beta_cholesky,_=torch.gesv(XtY,XtX)
		return beta_cholesky
if __name__=="__main__":
	model=UNet2D_4Lanes_LSF(None)
	x_map,y_map=generate_xy_coordinates(10,10,4)
	x_map=np.stack([x_map for i in range(2)])
	y_map=np.stack([y_map for i in range(2)])
	x_map=torch.FloatTensor(x_map).to("cuda")
	y_map=torch.FloatTensor(y_map).to("cuda")	
	model.forward(None,x_map,y_map)


		