
# coding: utf-8

# In[18]:


from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data, io
from skimage.color import label2rgb
import skimage
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


# In[80]:


# settings for LBP
radius = 6
n_points = 2 * radius


# In[81]:


# 读取图像
image = cv2.imread("/Users/sijian/Documents/Github/Fall2017-project3-fall2017-project3-grp6/data/training_set/images/img_0001.jpg")


# In[82]:


#显示到plt中，需要从BGR转化到RGB，若是cv2.imshow(win_name, image)，则不需要转化
image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(131)
plt.imshow(image1)
plt.show()


# In[83]:


# 转换为灰度图显示
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.subplot(132)
plt.imshow(image, cmap='gray')
plt.show()


# In[84]:


# 处理
lbp = local_binary_pattern(image, n_points, radius)
print lbp
plt.subplot(133)
plt.imshow(lbp, cmap='gray')
plt.show()


# In[85]:


print lbp.shape

