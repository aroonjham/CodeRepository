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
require(xgboost)

data <- Caravan

### PreProcessing

### There are a few predictors with only a single value, so we remove these first
isZV <- apply(data, 2, function(x) length(unique(x)) == 1)
data <- data[, !isZV]

data.x <- data[,-c(86)]
data.y <- as.data.frame(data[,c(86)])

segPP <- preProcess(data.x, c("BoxCox"))
data.x.tr <- predict(segPP, data.x)

segCorr <- cor(data.x.tr)

#library(corrplot)
#corrplot(segCorr, order = "hclust", tl.cex = .35)

highCorr <- findCorrelation(segCorr, .75)
data.x.tr.filter <- data.x.tr[, -highCorr]

nzv <- nearZeroVar(data.x.tr.filter)
data.final <- cbind(data.x.tr.filter[, -nzv],data.y)

colnames(data.final)[colnames(data.final) == 'data[, c(86)]'] <- 'Purchase'

### Split data between test and train

set.seed(123)
train= sample(c(TRUE,TRUE,TRUE,FALSE), nrow(data.x),rep=TRUE)
test = (!train)

data.train <- data.final[train,]
data.test <- data.final[test,]

### Logistic Regression

glm.fit=glm(Purchase ~ ., data = data.train, family=binomial)
summary(glm.fit)
glm.probs=predict(glm.fit,newdata=data.test,type="response") 

glm.ROCR = prediction(glm.probs, data.test$Purchase)
glm.perfROCR = performance(glm.ROCR, "tpr", "fpr")
plot(glm.perfROCR, colorize=TRUE)
performance(glm.ROCR, "auc")@y.values

glm.pred=ifelse(glm.probs >0.06,"Yes","No")
table(glm.pred,data.test$Purchase)

probseq <- seq(0.02,0.34,0.01)
tp.ax <- c()
fp.ax <- c()
tn.ax <- c()
fn.ax <- c()
x.ax <- c()
#tbl <- data.frame(x.ax = integer(), y.ax = integer())
for (i in probseq){
glm.pred=ifelse(glm.probs >i,"Yes","No")
df <- as.data.frame(table(glm.pred,data.test$Purchase))
tp.ax <- append(tp.ax, df[df$glm.pred=="Yes" & df$Var2=="Yes",c("Freq")])
fp.ax <- append(fp.ax, df[df$glm.pred=="Yes" & df$Var2=="No",c("Freq")])
fn.ax <- append(fn.ax, df[df$glm.pred=="No" & df$Var2=="Yes",c("Freq")])
tn.ax <- append(tn.ax, df[df$glm.pred=="No" & df$Var2=="No",c("Freq")])
x.ax <- append(x.ax,i)
}

plot(x.ax,(tp.ax/(tp.ax+fp.ax)), xlab = "Threshold", ylab = "Conversion rate", type = "l", col = "blue")
plot(tp.ax, fp.ax, xlab = "True Conversions", ylab = "False leads", type = "l", col = "blue")

### Linear Discriminant Analysis

lda.fit<-lda(Purchase ~ ., data = data.train)
plot(lda.fit)

lda.pred=predict(lda.fit,newdata=data.test) 
lda.pred.df <- as.data.frame(lda.pred$posterior)

lda.ROCR = prediction(lda.pred.df$Yes, data.test$Purchase)
lda.perfROCR = performance(lda.ROCR, "tpr", "fpr")
plot(lda.perfROCR, colorize=TRUE)
performance(lda.ROCR, "auc")@y.values

probseq <- seq(0.02,0.33,0.01)
tp.ax <- c()
fp.ax <- c()
tn.ax <- c()
fn.ax <- c()
x.ax <- c()
#tbl <- data.frame(x.ax = integer(), y.ax = integer())
for (i in probseq){
lda.prob=ifelse(lda.pred.df$Yes >i,"Yes","No")
df <- as.data.frame(table(lda.prob,data.test$Purchase))
tp.ax <- append(tp.ax, df[df$lda.prob=="Yes" & df$Var2=="Yes",c("Freq")])
fp.ax <- append(fp.ax, df[df$lda.prob=="Yes" & df$Var2=="No",c("Freq")])
fn.ax <- append(fn.ax, df[df$lda.prob=="No" & df$Var2=="Yes",c("Freq")])
tn.ax <- append(tn.ax, df[df$lda.prob=="No" & df$Var2=="Yes",c("Freq")])
x.ax <- append(x.ax,i)
}

plot(x.ax,(tp.ax/(tp.ax+fp.ax)), xlab = "Threshold", ylab = "Conversion rate", type = "l", col = "blue")
plot(tp.ax, fp.ax, xlab = "True Conversions", ylab = "False leads", type = "l", col = "blue")

### Quadratic Discriminant Analysis

qda.fit=qda(Purchase ~ ., data = data.train)

qda.pred=predict(qda.fit,newdata=data.test) 
qda.pred.df <- as.data.frame(qda.pred$posterior)

qda.ROCR = prediction(qda.pred.df$Yes, data.test$Purchase)
qda.perfROCR = performance(qda.ROCR, "tpr", "fpr")
plot(qda.perfROCR, colorize=TRUE)
performance(qda.ROCR, "auc")@y.values

probseq <- seq(0.02,0.95,0.01)
tp.ax <- c()
fp.ax <- c()
tn.ax <- c()
fn.ax <- c()
x.ax <- c()
#tbl <- data.frame(x.ax = integer(), y.ax = integer())
for (i in probseq){
qda.prob=ifelse(qda.pred.df$Yes >i,"Yes","No")
df <- as.data.frame(table(qda.prob,data.test$Purchase))
tp.ax <- append(tp.ax, df[df$qda.prob=="Yes" & df$Var2=="Yes",c("Freq")])
fp.ax <- append(fp.ax, df[df$qda.prob=="Yes" & df$Var2=="No",c("Freq")])
fn.ax <- append(fn.ax, df[df$qda.prob=="No" & df$Var2=="Yes",c("Freq")])
tn.ax <- append(tn.ax, df[df$qda.prob=="No" & df$Var2=="Yes",c("Freq")])
x.ax <- append(x.ax,i)
}

plot(x.ax,(tp.ax/(tp.ax+fp.ax)), xlab = "Threshold", ylab = "Conversion rate", type = "l", col = "blue")
plot(tp.ax, fp.ax, xlab = "True Conversions", ylab = "False leads", type = "l", col = "blue")

### Random Forest

rf.fit <- randomForest(Purchase ~., data = data.train, mtry = 8, ntree = 2000, importance =TRUE)

rf.probs=predict(rf.fit,data.test, type = "prob")
rf.probs.df <- as.data.frame(rf.probs)
rf.ROCR = prediction(rf.probs.df$Yes, data.test$Purchase)
rf.perfROCR = performance(rf.ROCR, "tpr", "fpr")
plot(rf.perfROCR, colorize=TRUE)
performance(rf.ROCR, "auc")@y.values[[1]]

varImpPlot(rf.fit)

probseq <- seq(0.02,0.84,0.01)
tp.ax <- c()
fp.ax <- c()
tn.ax <- c()
fn.ax <- c()
x.ax <- c()
#tbl <- data.frame(x.ax = integer(), y.ax = integer())
for (i in probseq){
rf.prob=ifelse(rf.probs.df$Yes >i,"Yes","No")
df <- as.data.frame(table(rf.prob,data.test$Purchase))
tp.ax <- append(tp.ax, df[df$rf.prob=="Yes" & df$Var2=="Yes",c("Freq")])
fp.ax <- append(fp.ax, df[df$rf.prob=="Yes" & df$Var2=="No",c("Freq")])
fn.ax <- append(fn.ax, df[df$rf.prob=="No" & df$Var2=="Yes",c("Freq")])
tn.ax <- append(tn.ax, df[df$rf.prob=="No" & df$Var2=="Yes",c("Freq")])
x.ax <- append(x.ax,i)
}

plot(x.ax,(tp.ax/(tp.ax+fp.ax)), xlab = "Threshold", ylab = "Conversion rate", type = "l", col = "blue")
plot(tp.ax, fp.ax, xlab = "True Conversions", ylab = "False leads", type = "l", col = "blue")

### Boost 

## need slight alteration to data frame as boost accepts only 0,1 as binomial factor
data.train$Purchase1 <- ifelse(data.train$Purchase == "Yes", 1, 0)
data.test$Purchase1 <- ifelse(data.test$Purchase == "Yes", 1, 0)

rem <- c("Purchase")
data.train <- data.train[,!(names(data.train) %in% rem)]
data.test <- data.test[,!(names(data.test) %in% rem)]
## model creation with no interaction
boost.fit <- gbm(Purchase1 ~ ., data = data.train, distribution= "bernoulli",n.trees =2000 , shrinkage =0.01, interaction.depth = 1)

boost.probs=predict(boost.fit,data.test, n.trees = 2000, type = "response")
boost.ROCR = prediction(boost.probs, data.test$Purchase1)
boost.perfROCR = performance(boost.ROCR, "tpr", "fpr")
plot(boost.perfROCR, colorize=TRUE)
performance(boost.ROCR, "auc")@y.values[[1]]

data.train$Purchase <- ifelse(data.train$Purchase1 == 1, "Yes", "No")
data.test$Purchase <- ifelse(data.test$Purchase1 == 1, "Yes", "No")

probseq <- seq(0.01,0.36,0.01)
tp.ax <- c()
fp.ax <- c()
tn.ax <- c()
fn.ax <- c()
x.ax <- c()
#tbl <- data.frame(x.ax = integer(), y.ax = integer())
for (i in probseq){
boost.prob=ifelse(boost.probs >i,"Yes","No")
df <- as.data.frame(table(boost.prob,data.test$Purchase))
tp.ax <- append(tp.ax, df[df$boost.prob=="Yes" & df$Var2=="Yes",c("Freq")])
fp.ax <- append(fp.ax, df[df$boost.prob=="Yes" & df$Var2=="No",c("Freq")])
fn.ax <- append(fn.ax, df[df$boost.prob=="No" & df$Var2=="Yes",c("Freq")])
tn.ax <- append(tn.ax, df[df$boost.prob=="No" & df$Var2=="Yes",c("Freq")])
x.ax <- append(x.ax,i)
}

plot(x.ax,(tp.ax/(tp.ax+fp.ax)), xlab = "Threshold", ylab = "Conversion rate", type = "l", col = "blue")
plot(tp.ax, fp.ax, xlab = "True Conversions", ylab = "False leads", type = "l", col = "blue")

summary(boost.fit)

'''''''''''''''''''''''''''''
## model creation with interaction
boost.fit <- gbm(Purchase1 ~ ., data = data.train, distribution= "bernoulli",n.trees =2000 , shrinkage =0.01, interaction.depth = 2)

boost.probs=predict(boost.fit,data.test, n.trees = 2000, type = "response")
boost.ROCR = prediction(boost.probs, data.test$Purchase1)
boost.perfROCR = performance(boost.ROCR, "tpr", "fpr")
plot(boost.perfROCR, colorize=TRUE)
performance(boost.ROCR, "auc")@y.values[[1]]

# alternate boost method not to be used in RStudio

# Number of folds
tr.control = trainControl(method = "cv", number = 10)
shrinkage = (1:10)*0.001
n.trees =2000
interaction.depth = 1
boost.grid = expand.grid(shrinkage = shrinkage, n.trees = n.trees, interaction.depth=interaction.depth)

boost.tr = train(Purchase1 ~.,data = data.train,method = "gbm",trControl = tr.control, tuneGrid = boost.grid)
boost.fit = boost.tr$finalModel

boost.probs=predict(boost.fit,data.test, n.trees = 2000, type = "response")
boost.ROCR = prediction(boost.probs, data.test$Purchase1)
boost.perfROCR = performance(boost.ROCR, "tpr", "fpr")
plot(boost.perfROCR, colorize=TRUE)
performance(boost.ROCR, "auc")@y.values[[1]]
'''''''''''''''''''''''''''''

### Lasso

# need slight alteration to data frame and separation of predictor and response variables 
data.train$Purchase1 <- ifelse(data.train$Purchase == "Yes", 1, 0)
data.test$Purchase1 <- ifelse(data.test$Purchase == "Yes", 1, 0)

rem <- c("Purchase")
data.train <- data.train[,!(names(data.train) %in% rem)]
data.test <- data.test[,!(names(data.test) %in% rem)]

x.train =model.matrix(Purchase1~.,data.train)[,-1]
y.train = data.train$Purchase1

x.test =model.matrix(Purchase1~.,data.test)[,-1]
y.test = data.test$Purchase1

#model building
lasso.mod =glmnet(x.train,y.train,alpha =1, nlambda = 500, family = "binomial")

set.seed(1)
cv.out =cv.glmnet(x.train,y.train,alpha =1)
plot(cv.out)
bestlam = cv.out$lambda.min #cv.out$lambda.1se
lasso.coef=predict(lasso.mod ,type ="coefficients",s=bestlam )[1:41 ,]
round(lasso.coef,2)

lasso.probs=predict(lasso.mod ,s=bestlam ,newx=x.test, type = "response")
lasso.ROCR = prediction(lasso.probs, data.test$Purchase1)
lasso.perfROCR = performance(lasso.ROCR, "tpr", "fpr")
plot(lasso.perfROCR, colorize=TRUE)
performance(lasso.ROCR, "auc")@y.values[[1]]

data.train$Purchase <- ifelse(data.train$Purchase1 == 1, "Yes", "No")
data.test$Purchase <- ifelse(data.test$Purchase1 == 1, "Yes", "No")

probseq <- seq(0.01,0.28,0.01)
tp.ax <- c()
fp.ax <- c()
tn.ax <- c()
fn.ax <- c()
x.ax <- c()
#tbl <- data.frame(x.ax = integer(), y.ax = integer())
for (i in probseq){
lasso.prob=ifelse(lasso.probs >i,"Yes","No")
df <- as.data.frame(table(lasso.prob,data.test$Purchase))
tp.ax <- append(tp.ax, df[df$lasso.prob=="Yes" & df$Var2=="Yes",c("Freq")])
fp.ax <- append(fp.ax, df[df$lasso.prob=="Yes" & df$Var2=="No",c("Freq")])
fn.ax <- append(fn.ax, df[df$lasso.prob=="No" & df$Var2=="Yes",c("Freq")])
tn.ax <- append(tn.ax, df[df$lasso.prob=="No" & df$Var2=="No",c("Freq")])
x.ax <- append(x.ax,i)
}

plot(x.ax,(tp.ax/(tp.ax+fp.ax)), xlab = "Threshold", ylab = "Conversion rate", type = "l", col = "blue")
plot(tp.ax, fp.ax, xlab = "True Conversions", ylab = "False leads", type = "l", col = "blue")

export.df <- cbind(tp.ax, fp.ax, fn.ax, tn.ax, x.ax)


### xgBoost

	data <- Caravan

	data$Purchase <- ifelse(data$Purchase == "Yes", 1, 0)

	data.x <- as.matrix(data[,-c(86)])
	data.y <- as.matrix(data[,c(86)])

	set.seed(123)
	train= sample(c(TRUE,TRUE,TRUE,FALSE), nrow(data.x),rep=TRUE)
	test = (!train)

	data.x.train <- data.x[train,]
	data.y.train <- data.y[train,]

	data.x.test <- data.x[test,]
	data.y.test <- data.y[test,]


## doing cross validation in xgboost

cv.mod <- xgb.cv(data = data.x.train, nfold = 5, verbose = T, label = data.y.train, nrounds = 700, 
					objective = 'binary:logistic', eval_metric = 'auc', eta = 0.01, max_depth = 10,
					gamma = 0.5, min_child_weight = 0.8)

					
cv.mod1 <- xgb.cv(data = data.x.train, nfold = 5, verbose = T, label = data.y.train, nrounds = 1200, 
					objective = 'binary:logistic', eval_metric = 'auc', eta = 0.003, max_depth = 15,
					gamma = 0.75, min_child_weight = 0.6, colsample_bytree = 0.6, subsample = 0.6)					
					
plot(cv.mod$test.auc.mean)
lines(cv.mod1$test.auc.mean)

## build model and predict
					
xgbmod <- xgboost(data = data.x.train, label = data.y.train, nrounds = 700, 
					objective = 'binary:logistic', eval_metric = 'auc', eta = 0.01, max_depth = 10,
					gamma = 0.5, min_child_weight = 0.8)

importance_matrix <- xgb.importance(colnames(data.x.train), model = xgbmod)
xgb.plot.importance(importance_matrix) #Ckmeans.1d.dp package required
					
testpred <- predict(xgbmod, data.x.test)			

# calculate auc for the model

xgb.ROCR = prediction(testpred, data.y.test)
xgb.perfROCR = performance(xgb.ROCR, "tpr", "fpr")
plot(xgb.perfROCR, colorize=TRUE)
performance(xgb.ROCR, "auc")@y.values




################################################################################################


svm.fit <- svm(Purchase ~., data = data.train, kernel ="radial", gamma =1, cost =0.01, decision.values =T)
sv.pred <- predict(svm.fit, data.test, decision.values =T)
sv.pred <- attributes(predict(svm.fit, data.test, decision.values =T))$decision.values
sv.ROCR = prediction(sv.pred, data.test$Purchase)
sv.perfROCR = performance(sv.ROCR, "tpr", "fpr")
plot(sv.perfROCR, colorize=TRUE)
performance(sv.ROCR, "auc")@y.values[[1]]


svm.fit <- svm(Purchase ~., data = data.train, kernel ="linear",  cost =0.001, decision.values =T)
sv.pred <- predict(svm.fit, data.test, decision.values =T)
sv.pred <- attributes(predict(svm.fit, data.test, decision.values =T))$decision.values
sv.ROCR = prediction(sv.pred, data.test$Purchase)
sv.perfROCR = performance(sv.ROCR, "tpr", "fpr")
plot(sv.perfROCR, colorize=TRUE)
performance(sv.ROCR, "auc")@y.values[[1]]


svm.fit <- svm(Purchase ~., data = data.train, kernel ="radial", gamma =1, cost =1)
plot(svm.fit , data.train)

set.seed(1)
tune.out=tune(svm , Purchase~., data=data.train, kernel ="linear", ranges =list(cost=c(0.1 ,1 ,10 ,100 ,1000)))

