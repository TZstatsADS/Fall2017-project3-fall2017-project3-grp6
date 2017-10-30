library(sgd)
library(MASS)
#setwd("D:/Fall 2017/ADS/Fall2017-project3-fall2017-project3-grp6-master/data")
hog <- read.csv(file="hog3000.csv",head = T, sep = ",")
lbp <- read.csv(file="lbp.csv",head = F, sep = ",")
#sift <- read.csv(file="sift_train.csv",head = T)
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
knn.pred= knn(training[,-1], testing[,-1], cl= training$y, k = 8, l = 1)
mean(testing$y==knn.pred)             ##60% accuracy


####################################################
############Logistic (multinomial)!!!!!!!!!!
library(nnet)
glm.fit=multinom(y~., data=training,trace = F)
glm=multinom(y~., data=training,maxit = 500,trace = F)
summary(glm.fit)
#Prediction
glm.pred<-predict(glm.fit, testing, type = "class")
mean(testing$y==glm.pred)  ##82%

#cv
cv.lr <- function(X.train, y.train,K){
  n <- length(y.train)
  n.fold <- floor(n/K)
  s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
  cv.accu.lr <- rep(NA, K)
  
  for (i in 1:K){
    train.data <- X.train[s != i,]
    train.label <- y.train[s != i]
    test.data <- X.train[s == i,]
    test.label <- y.train[s == i]

    glm=multinom(train.label~., data=train.data,maxit = 500,trace = F)

    #Prediction
    pred<-predict(glm, test.data, type = "class")
    cv.accu.lr[i] <- mean(pred == test.label)  
    
  }			
  print(mean(cv.accu.lr))
  
}

cv.lr(training[,-1],training[,1], 5)

################################################# xgboost
library(readr)
library(xgboost)

trainnn<-as.matrix(training)
testtt<-as.matrix(testing)
dtrain=xgb.DMatrix(data=trainnn[,-1],label=trainnn[,1])

#bst=xgb.train(data=dtrain,params = param,nrounds=100)

#NROUNDS = c(100,300,500,1000)
#ETA = c(0.01, 0.1, 0.2, 0.3, 0.5, 0.8)
#MAX_DEPTH = c(6, 8, 10, 15, 20)
NROUNDS = c(500,1000)
ETA = c(0.3)
MAX_DEPTH = c(3,4,5,6)

cv.xgb <- function(X.train, y.train, K, NROUNDS, ETA, MAX_DEPTH){
  for (nround in NROUNDS){
    for (eta in ETA){
      for (max_depth in MAX_DEPTH){
        n <- length(y.train)
        n.fold <- floor(n/K)
        s <- sample(rep(1:K, c(rep(n.fold, K-1), n-(K-1)*n.fold)))  
        cv.acc <- rep(NA, K)
        
        for (i in 1:K){
          train.data <- X.train[s != i,]
          train.label <- y.train[s != i]
          test.data <- X.train[s == i,]
          test.label <- y.train[s == i]
          
          param <- list("objective" = "multi:softmax",
                        "eval_metric" = "mlogloss",
                        "num_class" = 3, 'eta' = eta, 'max_depth' = max_depth)
          
          dtrain=xgb.DMatrix(data=train.data,label=train.label)
          
          bst <- xgb.train(data = dtrain,  param = param, nrounds = nround)
          pred <- predict(bst, newdata = test.data) 
          
          cv.acc[i] <- mean(pred == test.label)  
        }			
        print(paste("------Mean 5-fold cv accuracy for nround=",nround,",eta=",eta,",max_depth=",max_depth,
                    "------",mean(cv.acc)))
        key = c(nround,eta,max_depth)
        CV_ERRORS[key] = mean(cv.acc)
      
      }
    }
  }
}

CV_ERRORS = list()
cv.xgb(trainnn[,-1], trainnn[,1], 5, NROUNDS, ETA, MAX_DEPTH)
#############eta=0.3,max_depth=4,nround=1000
param <- list("objective" = "multi:softmax",
              "eval_metric" = "mlogloss",
              "num_class" = 3, 'eta' = 0.3, 'max_depth' = 4)

dtrain=xgb.DMatrix(data=trainnn[,-1],label=trainnn[,1])
#############Final Model:
bst <- xgb.train(data = dtrain,  param = param, nrounds = 1000)
# Predict hold-out test set
test_pred <- predict(bst, newdata = testtt[,-1])

mean(test_pred==testing$y)
####hog+xgboost=79.5%
