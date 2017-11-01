# Project: Dogs, Fried Chicken or Blueberry Muffins?
![image](figs/chicken.jpg)
![image](figs/muffin.jpg)

### [Full Project Description](doc/project3_desc.md)

Term: Fall 2017

+ Team #6
+ Team members
	+ Sijian Xuan
	+ Xinyao Guo
	+ Siyi Wang
	+ Pinren Chen
	+ Xiaoyu Zhou

+ Project summary: In this project, we used several feature extraction such as LBP feature extraction, HoG feature extraction methods and classification models(SVM, BPNN, Random Forest, Xgboost, Logistic Regression, GBM) from machine learning to recogonize whether there is a dog, chicken or blueberry muffin in the image. We have a baseline model which is GBM + SIFT and we are trying to develop a way that improve the baseline model most. We also tried RGB for feature extraction and CNN for both feature extraction and classification. However, there are grayscale images that RGB could not deal with; and CNN takes a long time to train the model. We finally use SVM + LBP and xgboost + HoG as our winners. They achieve a accuracy rate of about 80% and takes a short time to train (less than 1 minute).
	
**Contribution statement**: 
Sijian Xuan: as the group presenter, is working on the whole organization of the study. He does the research about LBP feature extraction method and choose BPNN, SVM, Random Forest, Logistic regression as well as writing the relevant code with help of Siyi Wang and Xinyao Guo. He collects everyone's code and write them in main.Rmd and write the ppt file.

Xinyao Guo: tried HoG and RGB to do feature extraction and finally chose HoG. Applied logistic regression(multinomial), random forest, KNN, xgboost as candidate classification models. Tune the models by grid searching method. Discard KNN as the accuracy is only around 60%. Applied cross validation to prevent the overfitting issue.

Xiyi Wang:

Pinren Chen: Responsible for CNN model analysis. Conducted CNN in python for both feature extraction and classification. Debuged and improved model with Xiaoyu. Saved the model and used it for prediction.

Xiaoyu Zhou: was responsible for CNN model analysis. Conducted CNN in R for both feature extraction and classification, decreased the baseline error rate successfully but discarded R-version due to the time spend. Collaborated with Pinren to write code and debug in Python for CNN feature extraction and classification, and achieved the accuracy rate >90%. Accelerated the Python running time.


Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
