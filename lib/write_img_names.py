
# coding: utf-8
# author: Sijian Xuan

import numpy as np

a = np.arange(1, 3001, 1)
name = [None]*3000

for a in range(1,3001):
    if a < 10:
        name[a-1] = 'img_000' + str(a) + '.jpg'
    elif a < 100:
        name[a-1] = 'img_00' + str(a) + '.jpg'
    elif a < 1000:
        name[a-1] = 'img_0' + str(a) + '.jpg'
    else:
        name[a-1] = 'img_' + str(a) + '.jpg'

thefile = open('../output/train_img_names.txt', 'w')

for item in name:
    thefile.write("%s\n" % item)

