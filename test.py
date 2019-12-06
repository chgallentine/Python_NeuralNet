# -*- coding: utf-8 -*-
# @Author: Charlie Gallentine
# @Date:   2019-11-28 11:48:02
# @Last Modified by:   Charlie Gallentine
# @Last Modified time: 2019-11-28 19:42:36
import numpy as np 

a = np.ones([1,5])
b = np.zeros([1,3])

for i,val in enumerate(b[0]):
	a[0][i] = val

print(a)