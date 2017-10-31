# using HOG to get features
#install.packages("OpenImageR")
library(OpenImageR)

hog <- vector()

for (i in 1:3000){
  n<-nchar(as.character(i))
  path<-paste0("../images/img_",paste(rep(0,4-n),collapse=""),i,".jpg")
  a <- readImage(path)
  hog <- rbind(hog,HOG(a))
}
write.csv(hog,file = "../output/hog_feature.csv")