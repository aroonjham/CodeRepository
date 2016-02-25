# install packages using install.packages("glmnet")

# load libraries
require(psych)
require(neuralnet)
require(MASS)
require(ISLR)
require(randomForest)
require(gbm) ### for boosting
require(boot)
require(glmnet) ### for lasso and ridge
require(forecast)
require(rpart)
require(rpart.plot)
require(e1071)
require(ROCR)
require(class)#knn
require(caret)
require(AppliedPredictiveModeling)

#clean data
caravan.train <- Caravan[1:1000,]
caravan.test <- Caravan[1001:5822,]
caravan.train$Purchase1 <- ifelse(caravan.train$Purchase == "Yes", 1, 0)
caravan.test$Purchase1 <- ifelse(caravan.test$Purchase == "Yes", 1, 0)

#run boost
boost.caravan <- gbm(Purchase1 ~ .-Purchase, data = caravan.train, distribution=
"bernoulli",n.trees =1000 , shrinkage =0.01)
summary(boost.caravan)

caravan.probs=predict(boost.caravan,newdata=caravan.test,n.trees =1000, type="response") 
caravan.pred=ifelse(caravan.probs >0.2,"Yes","No")
table(caravan.pred,caravan.test$Purchase)

#run logistic regression
glm.fit=glm(Purchase1 ~ .-Purchase, data = caravan.train, family=binomial)
glm.probs=predict(glm.fit,newdata=caravan.test,type="response") 
glm.pred=ifelse(glm.probs >0.20,"Yes","No")
table(glm.pred,caravan.test$Purchase)

probseq <- seq(0.05,0.90,0.01)
tp.ax <- c()
fp.ax <- c()
x.ax <- c()
#tbl <- data.frame(x.ax = integer(), y.ax = integer())
for (i in probseq){
glm.pred=ifelse(glm.probs >i,"Yes","No")
df <- as.data.frame(table(glm.pred,data.test$default))
tp.ax <- append(tp.ax, df[df$glm.pred=="Yes" & df$Var2=="Yes",c("Freq")])
fp.ax <- append(fp.ax, df[df$glm.pred=="Yes" & df$Var2=="No",c("Freq")])
x.ax <- append(x.ax,i)
}

#run linear discriminant analysis
#lda.fit=lda(Purchase1 ~ .-Purchase, data = caravan.train)

#k nearest neighbors
knn.caravan.train <- Caravan[1:1000,]
knn.caravan.test <- Caravan[1001:5822,]
knn.caravan.train$Purchase <- ifelse(caravan.train$Purchase == "Yes", 1, 0)
knn.caravan.test$Purchase <- ifelse(caravan.test$Purchase == "Yes", 1, 0)
knn.pred=knn(knn.caravan.train, knn.caravan.test, knn.caravan.train$Purchase, k=1)
table(knn.pred,knn.caravan.test$Purchase)

#example of random forest
set.seed(1)
mat <- matrix(, nrow = 100, ncol = 50)
for (i in 1:50) {
mat[,i] <- rnorm(100)
}
require(randomForest)
rf.mat =randomForest(V1~.,data=mat, mtry=7, importance =TRUE)
importance(rf.mat)

#example of neural network

norm.fun = function(x){(x - min(x))/(max(x) - min(x))}
data.norm = apply(Boston, 2, norm.fun) #apply normalization function over Boston data

train.data.norm <- data.norm[1:306,]
test.data.norm <- data.norm[307:506,]

net <- neuralnet(medv ~ crim + zn + indus + chas + nox   + rm  +  age + dis  +  rad + tax  + ptratio +  black  + lstat
, data = train.data.norm, hidden = 2, threshold = 0.001, rep=2) 

print(net)
plot(net)

net.test<-compute(net,test.data.norm[,1:13])

net.testouputdata<-net.test$net.result
net.top <- as.data.frame(net.testouputdata)
net.top <- cbind(net.top, test.data.norm[,14])
net.toperror <- mean((net.top[,1] - test.data.norm[,14])^2)
net.toperror

#same example above with random forest

rf.test <- randomForest(medv~.,data=train.data.norm,mtry=4,ntree=400)
rf.testouputdata=predict(rf.test,test.data.norm)

rf.top <- as.data.frame(rf.testouputdata)
rf.top <- cbind(rf.top, test.data.norm[,14])

rf.toperror <- mean((rf.top[,1] - test.data.norm[,14])^2)
rf.toperror
importance(rf.test)

#same example above with boosting
train.data.norm <- as.data.frame(train.data.norm)
test.data.norm <- as.data.frame(test.data.norm)

boost.boston=gbm(medv~.,data=train.data.norm,distribution="gaussian",n.trees=10000,shrinkage=0.01,interaction.depth=4) # if shrinkage is made smaller, more trees will be needed
summary(boost.boston) #lists importance of key predictor variables

boost.testouputdata=predict(boost.boston,test.data.norm,n.trees=10000)

boost.top <- as.data.frame(boost.testouputdata)
boost.top <- cbind(boost.top, test.data.norm[,14])

boost.toperror <- mean((boost.top[,1] - test.data.norm[,14])^2)
boost.toperror


## code for Lasso
data = Boston
data = Boston
train= sample(c(TRUE,TRUE,FALSE), nrow(data),rep=TRUE)
test = (!train)
#lasso needs to separate predictor variables (x) from response variable (y)
x=model.matrix(medv~.,data)[,-1]
y = data$medv

grid =10^seq(10,-2, length =100)
lasso.mod =glmnet(x[train,],y[train],alpha =1, lambda =grid) #build a lasso model using training set for lambda values ranging from 10^10 to 10^-2

set.seed(1)
cv.out =cv.glmnet(x[train,],y[train],alpha =1)
plot(cv.out)
bestlam = cv.out$lambda.min #cv.out$lambda.1se
lasso.coef=predict(lasso.mod ,type ="coefficients",s=bestlam )[1:14 ,]
round(lasso.coef,1)

lasso.pred=predict(lasso.mod ,s=bestlam ,newx=x[test,])
compare <- cbind(lasso.pred, y.test)
mse <- mean((lasso.pred - y[test])^2)

SSE = sum((lasso.pred - y[test])^2)
SST = sum((mean(y) - y[test])^2)
R2 = 1 - (SSE/SST)

### using lasso for classification problem

data <- Caravan
set.seed(123)
train= sample(c(TRUE ,FALSE), nrow(data),rep=TRUE)
train= sample(c(TRUE ,FALSE), nrow(data),rep=TRUE)
test = (!train)
data$Purchase1 <- ifelse(data$Purchase == "Yes", 1, 0)

rem <- c("Purchase")
data <- data[,!(names(data) %in% rem)]

x=model.matrix(Purchase1~.,data)[,-1]
y = data$Purchase1
lasso.mod =glmnet(x[train,],y[train],alpha =1, nlambda = 500)

set.seed(1)
cv.out =cv.glmnet(x[train,],y[train],alpha =1)
plot(cv.out)
bestlam = cv.out$lambda.min #cv.out$lambda.1se
lasso.coef=predict(lasso.mod ,type ="coefficients",s=bestlam )[1:84 ,]
round(lasso.coef,1)

lasso.pred=predict(lasso.mod ,s=bestlam ,newx=x[test,])
lasso.predict=ifelse(lasso.pred >0.2,1,0)
table(lasso.predict, y[test])

## compare to boost
boost.caravan <- gbm(Purchase1 ~ ., data = data[train,], distribution=
"bernoulli",n.trees =1000 , shrinkage =0.01)
summary(boost.caravan)

caravan.probs=predict(boost.caravan,newdata=data[test,],n.trees =1000, type="response") 
caravan.pred=ifelse(caravan.probs >0.2,"Yes","No")
data.test <- data[test,]
table(caravan.pred,data.test$Purchase1)


####### trying something new #############


data.x.pp <- preProcess(Caravan[,-c(86)], method = c("center", "scale")) ##use Caravan data under ISLR. Standardize all predictor values
data.x <- predict(data.x.pp,Caravan[,-c(86)]) ##apply transform to the data
data.x <- cbind(data.x,Caravan[,c(86)])
colnames(data.x)[colnames(data.x) == 'Caravan[, c(86)]'] <- 'Purchase'

set.seed(123)
train= sample(c(TRUE,TRUE,TRUE,FALSE), nrow(data.x),rep=TRUE)
test = (!train)

data.train <- data.x[train,]
data.test <- data.x[test,]