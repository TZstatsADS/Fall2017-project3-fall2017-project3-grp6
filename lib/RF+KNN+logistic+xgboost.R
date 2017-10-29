library(sgd)
library(MASS)
setwd("D:/Fall 2017/ADS/Fall2017-project3-fall2017-project3-grp6-master/data")
hog <- read.csv(file="hog3000.csv",head = T, sep = ",")
lbp <- read.csv(file="lbp.csv",head = F, sep = ",")
sift<-read.csv(file="sift_train.csv",head = T)
labels<-read.csv("label_train.csv")
y = labels[,2]
dat <- data.frame(y=y,x = lbp)


#dat_sift <-data.frame(y=y,x = sift) ###########can not run RF on sift features(too many predictors!!!)

splitdf <- function(dataframe, seed=NULL) {
  if (!is.null(seed)) set.seed(seed)
  index <- 1:nrow(dataframe)
  trainindex <- sample(index, trunc(length(index)*0.8))
  trainset <- dataframe[trainindex, ]
  testset <- dataframe[-trainindex, ]
  list(trainset=trainset,testset=testset)
}

splits <- splitdf(dat, seed = 100)
lapply(splits, nrow)
training <- splits$trainset
testing <- splits$testset
training <- training[,-2]
testing <- testing[,-2]
#############################################random forest
library(randomForest)
model <- randomForest(y~., 
                      data = training,
                      ntree=500,
                      importance=TRUE,
                      keep.forest=TRUE
)
print(model)
varImpPlot(model, type=1)
predicted <- predict(model, newdata=testing[ ,-1])
predicted <- round(predicted)
mean(predicted == testing$y)  ###
###hog+rf=66.2%
###
train.rf<- function(traindata) {
  
  traindata$y<- as.factor(traindata$y)
  y.index<- which(colnames(traindata)=="y")
  bestmtry <- tuneRF(y= traindata$y, x= traindata[,-y.index], stepFactor=1.5, improve=1e-5, ntree=600)
  best.mtry <- bestmtry[,1][which.min(bestmtry[,2])]
  
  
  model.rf <- randomForest(y ~ ., data = traindata, ntree=600, mtry=best.mtry, importance=T)
  return(model.rf)
}
test.rf <- function(model,test.data) {
  return(predict(model, test.data, type = "class"))
}
rf.model <- train.rf(training)
rf.pre=test.rf(rf.model,testing)
mean(rf.pre==testing$y)

# Knn
##########################################
library(class)
knn.pred= knn(training[,-1], testing[,-1], cl= training$y, k = 6, l = 1)
mean(testing$y==knn.pred)             ##60% accuracy


####################################################
############Logistic (multinomial)!!!!!!!!!!
library(nnet)
glm.fit=multinom(y~., data=training)
summary(glm.fit)
#Prediction
glm.pred<-predict(glm.fit, testing,type = "class")
mean(testing$y==glm.pred)  ##82%
################################################# xgboost
library(readr)
library(xgboost)

param <- list("objective" = "multi:softmax",
                   "eval_metric" = "mlogloss",
                   "num_class" = 3)
trainnn<-as.matrix(training)
testtt<-as.matrix(testing)
dtrain=xgb.DMatrix(data=trainnn[,-1],label=trainnn[,1])
a=xgb.cv(data=dtrain,params = param,nfold=5,nrounds=3000)
bst=xgb.train(data=dtrain,params = param,nrounds=3000)
#min.loss.idx = which.min(a$evaluation_log$test_mlogloss_mean) 
#cat ("Minimum logloss occurred in round : ", min.loss.idx, "\n")
#bst=xgboost(data=dtrain,params = param,nrounds=min.loss.idx)

# Predict hold-out test set
test_pred <- predict(bst, newdata = testtt[,-1])

mean(test_pred==testing$y)
####hog+xgboost=80%
####lbp+xgboost=76%