####Make LBP dataset
if(!require("EBImage")){
  source("https://bioconductor.org/biocLite.R")
  biocLite("EBImage")
}

packages.used=c("gbm", "caret","DMwR" ,"nnet","randomForest","e1071","data.table","readr","xgboost")
packages.needed=setdiff(packages.used, 
                        intersect(installed.packages()[,1], 
                                  packages.used))
if(length(packages.needed)>0){
  install.packages(packages.needed, dependencies = TRUE)
}

library("EBImage")
library("gbm")
library("caret")
library("DMwR")
library("nnet")
library("randomForest")
library("e1071")
library("data.table")
library("xgboost")
library("readr")

source("../lib/train.r")
source("../lib/test.r")
###########################这个是train的东西 课前先run好
sift.feature=read.csv("../data/sift_feature.csv", header = T)
lbp.feature=read.csv("../data/lbp_feature.csv", header = F)
hog.feature = read.csv("../data/hog_feature.csv")
label=read.csv("../data/trainlabel.csv")
####Make dataset
####sift
siftdata=data.frame(cbind(label,sift.feature[,-1]))
test.index=sample(1:3000,500,replace=F)
colnames(siftdata)[2]="y"
siftdata = siftdata[,-1]
test.sift = siftdata[test.index,]
test.x.sift = test.sift[,-1]
train.sift = siftdata[-test.index,]

####lbp
lbpdata = data.frame(cbind(label,lbp.feature))
colnames(lbpdata)[2] = "y"
lbpdata = lbpdata[,-1]
test.lbp = lbpdata[test.index,]
test.x.lbp = test.lbp[,-1]
train.lbp = lbpdata[-test.index,]

####hog
hogdata = data.frame(cbind(label,hog.feature[,-1]))
colnames(hogdata)[2] = "y"
hogdata = hogdata[,-1]
test.hog = hogdata[test.index,]
test.x.hog = test.hog[,-1]
train.hog = hogdata[-test.index,]

##############################################gbm model###之前先train好
gbm.model <- train.gbm(train.sift)
###########################################train SVM
svm.model <- train.svm(train.lbp)
#######################################train Hog
xgboost.model <- train.xgboost(train.hog)
#######################################time usage
b = system.time(gbm <- train.gbm(siftdata))
e = system.time(svm <- train.svm(lbpdata))
g = system.time(xgboost <- train.xgboost(hogdata))
time = list(gbm=b, svm=e, xgboost=g)
print(time)

########################################train part finished, next we extract feature and do test, have error rate and time
###################################################在班里现场从这里开始！！！
##########################################TEST part begins here
#########################################read feature(现场替换文件名称！！！！)
sift.feature=read.csv("../data/sift_feature.csv", header = T)
lbp.feature=read.csv("../data/lbp_feature.csv", header = F)
hog.feature = read.csv("../data/hog_feature.csv")
label=read.csv("../data/trainlabel.csv")
#SIFT feature
####Make dataset
sift_data=data.frame(cbind(label,sift.feature[,-1]))##这里label有两列，feature有5001列，多了两列名字
test.index = 1:n######################################替换n为图片数量！！！！！！！
colnames(sift_data)[2]="y"
sift_data = sift_data[,-1]##########################去掉label的名字
test.sift=sift_data
test.x.sift=test.sift[,-1]
####Make LBP dataset
lbpdata = data.frame(cbind(label,lbp.feature))##这里的lbpfeature只有59维，没有名字； label有名字
colnames(lbpdata)[2] = "y"
lbpdata = lbpdata[,-1]##########################去掉label的名字
test.lbp = lbpdata
test.x.lbp = test.lbp[,-1]
####Make HoG dataset
hogdata = data.frame(cbind(label,hog.feature[,-1]))###hog feature有名字，label有名字
colnames(hogdata)[2] = "y"
hogdata = hogdata[,-1]##########################去掉label的名字
test.hog = hogdata
test.x.hog = test.hog[,-1]

#####################################test SVM
svm.pre=test.svm(svm.model,test.x.lbp)
table(svm.pre,test.lbp$y)
mean(svm.pre != test.lbp$y)

####################################test HoG
xgboost.pre = test.xgboost(xgboost.model,test.hog)
table(xgboost.pre, test.hog$y)
mean(xgboost.pre != test.hog$y) 

########################################test GBM
gbm.pre = test.gbm(gbm.model,test.x.sift)
table(gbm.pre, test.sift$y)
mean(gbm.pre != test.sift$y) 



