#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : font_size19-04-21 12:02:31
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from matplotlib import pyplot as plt
import pandas as pd

epoch_train_path="E:/cv-adas/out-driver_161_90frame-scnn2d/epoch_train_loss.csv"
epoch_val_path="E:/cv-adas/out-driver_161_90frame-scnn2d/epoch_val_loss.csv"
iter_path="E:/cv-adas/out-driver_161_90frame-scnn2d/iter_loss.csv"

font_size=15
epoch_train_df=pd.read_csv(epoch_train_path,header=0)
epoch_val_df=pd.read_csv(epoch_val_path,header=0)
iter_df=pd.read_csv(iter_path,header=0)


fig=plt.figure(figsize=(15,4))
ax1=plt.subplot2grid((1,3),(0,0),colspan=2)
plt.title("Scnn-2d",fontsize=font_size)
plt.xlim(0,len(iter_df)+10)
plt.ylim(iter_df["loss"].min()-0.05,iter_df["loss"].max()+0.05)
plt.plot(iter_df["iteration"],iter_df["loss"],c="b")
plt.ylabel("BCE Loss",fontsize=font_size)
plt.xlabel("Iteration",fontsize=font_size)

ax2=plt.subplot2grid((1,3),(0,2))

plt.title("Scnn-2d",fontsize=font_size)
plt.plot(epoch_train_df["epochs"],epoch_train_df["loss"],c="b",label="train_loss",lw=2)
plt.plot(epoch_val_df["epochs"],epoch_val_df["loss"],c="r",ls="-",label="val_loss",lw=2)
#plt.ylabel("BCE Loss",fontsize=font_size)
plt.xlabel("Epoch",fontsize=font_size)
plt.legend()
plt.show()



