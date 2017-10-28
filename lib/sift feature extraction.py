
# coding: utf-8

# In[14]:


import cv2 
import numpy as np
from matplotlib import pyplot as plt


# In[67]:


def getSift(a):  
    b = str(a/10000.0)
    ch = ''
    n = 2
    b = n * ch + b[2:]
    img_path1 = "/Users/sijian/Documents/Github/Fall2017-project3-fall2017-project3-grp6/data/training_set/images/img_"+str(b)+".jpg"
    img = cv2.imread(img_path1)  
    gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(gray,None)  
    print type(kp),type(kp[0])  
    print kp[0].pt  
    des = sift.compute(gray,kp)  
    print type(kp),type(des)  
    print type(des[0]), type(des[1])  
    print des[0],des[1]  
    print des[1].shape  
    img=cv2.drawKeypoints(gray, kp, img, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(img),plt.show()


# In[69]:


getSift(6)

