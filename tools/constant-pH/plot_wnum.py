#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import pandas as pd
import statistics
import sys
import csv

import os

#####################################################################################
fontsize = 8
outputwidth = 7
outputheight = 3.9375

#####################################################################################

def moving_average(arr, window_size):
	kernel = np.ones(window_size) / window_size
	return np.convolve(arr, kernel, mode='valid')

#####################################################################################
fig = plt.figure()
ax  = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

#####################################################################################

colors = ['#DA291C','#56A8CB','#53A567']
	

#####################################################################################


df = pd.read_csv("zs-cs.csv")
zs = df.zs
cs = df.cs
			

#####################################################################################		
ax.plot(zs, cs, color="blue")

#####################################################################################
df.sort_values(by=["zs"],inplace=True)
cs = df.cs
zs = df.zs

cs_avg = moving_average(cs,1000)
zs_avg = moving_average(zs,1000)
ax2.plot(zs_avg,cs_avg,color="red")

#####################################################################################

plt.xlabel("z (A)")
plt.ylabel("c (#)")
plt.legend()

#####################################################################################

fig.tight_layout()
plt.subplots_adjust(top=0.95,bottom=0.15,left=0.1,right=0.95,hspace=0.1,wspace=0.15)

#####################################################################################
fileName = "zs-cs"
fig.set_size_inches(outputwidth,outputheight)
fig.savefig(fileName+".svg",dpi=300)

fig.set_size_inches(outputwidth,outputheight)
fig.savefig(fileName+".tif",dpi=300)
