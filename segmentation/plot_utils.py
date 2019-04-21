#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-04-21 12:02:31
# @Author  : Mengji Zhang (zmj_xy@sjtu.edu.cn)

from matplotlib import pyplot as plt
import pandas as pd

epoch_train_path="E:/cv-adas/out-driver_161_90frame-unet2d/epoch_train_loss.csv"
epoch_val_path="E:/cv-adas/out-driver_161_90frame-unet2d/epoch_val_loss.csv"

epoch_train_df=pd.read_csv(epoch_train_path,header=0)
epoch_val_df=pd.read_csv(epoch_val_path,header=0)

fig=plt.figure()
fig.add_subplot(111)
plt.plot(epoch_train_df["epochs"],epoch_train_df["loss"],c="b",label="train_loss",lw=2)
plt.plot(epoch_val_df["epochs"],epoch_val_df["loss"],c="r",ls="-",label="val_loss",lw=2)
plt.ylabel("BCE Loss",fontsize=20)
plt.xlabel("Epoch",fontsize=20)
plt.legend()
plt.show()


iter_path="E:/cv-adas/out-driver_161_90frame-unet2d/iter_loss.csv"
iter_df=pd.read_csv(iter_path,header=0)
fig=plt.figure(figsize=(10,5))
fig.add_subplot(111)
plt.xlim(0,len(iter_df)+10)
plt.plot(iter_df["iteration"],iter_df["loss"],c="b")
plt.ylabel("BCE Loss",fontsize=20)
plt.xlabel("iteration",fontsize=20)
plt.show()