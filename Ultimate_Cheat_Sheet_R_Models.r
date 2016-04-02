## R model cheat-sheet

################################### simple linear model #############################
library(AppliedPredictiveModeling)
data(solubility)

trainingData <- solTrainXtrans
trainingData$Solubility <- solTrainY

lmFitAllPredictors <- lm(Solubility ~ ., data = trainingData)

summary(lmFitAllPredictors)

lmPred1 <- predict(lmFitAllPredictors, solTestXtrans)

lmValues1 <- data.frame(obs = solTestY, pred = lmPred1)

defaultSummary(lmValues1) ## defaultSummary is a built in caret function to estimate the test set performance


################################### linear model using cross validation #############################

ctrl <- trainControl(method = "cv", number = 10) ### 10 fold cross validation

set.seed(100)
lmFit1 <- train(x = solTrainXtrans, y = solTrainY, method = "lm", trControl = ctrl)

## plot the residuals versus the predicted values AND predicted values versus the observed values

xyplot(solTrainY ~ predict(lmFit1), type = c("p", "g"), xlab = "Predicted", ylab = "Observed")
xyplot(resid(lmFit1) ~ predict(lmFit1), type = c("p", "g"), xlab = "Predicted", ylab = "Residuals")


################################### robust linear regression model #############################

set.seed(100)

# for rlm pre-process the predictors using PCA. rlm does not allow the covariance matrix of the predictors to be singular/
# A square matrix is singular, that is, its determinant is zero ... i.e. it contains rows or columns which are proportionally interrelated (highly correlated)

rlmPCA <- train(solTrainXtrans, solTrainY, method = "rlm",preProcess = "pca", trControl = ctrl)

################################### Partial Least Squares #############################

# predictors can be correlated
# if so ordinary least squares solution for multiple linear regression will have high variability and will become unstable
# solution is removal of the highly correlated predictors 
# However, this process does not necessarily ensure that linear combinations of predictors are uncorrelated.

# option 2: Using PCA for pre-processing guarantees that the resulting predictors, or combinations thereof, will be uncorrelated
# PCA results in new predictors are linear combinations of the original predictors, and thus, the practical understanding of the new predictors can become murky

# enter PLS. PLS iteratively seeks to find underlying, or latent, relationships among the predictors which are highly correlated with the response
# the tuning parameter here is "how many predictors to use". Its important to scale and center predictors

#step1 build model
plsTune <- train(solTrainXtrans, solTrainY, method = "pls", tuneLength = 20, trControl = ctrl, preProc = c("center", "scale"))

# step 2 finalize model
ncomp = as.integer(plsTune$bestTune)
plsPred <- predict(plsFit, solTestXtrans, ncomp = ncomp)

# step 3 measure effectiveness of the model
plsValues1 <- data.frame(obs = solTestY, pred = as.vector(plsPred))
defaultSummary(plsValues1)

################################### Elastic Net #############################

# 2 tuning parameters
# lambda adds a penalty on the sum of the squared regression parameters (ridge regression) L2 second order penalty
# fraction represents fraction of the full solution (Lasso regression). L1 first order penalty
# elastic net combines both the penalties

# step 0: create tuning grid
enetGrid <- expand.grid(.lambda = seq(0, 0.1, length = 5),.fraction = seq(.05, 1, length = 20))
#setting lambda to zero and tuning fraction gives us lasso. 
#Setting fraction to 1 and tuning lambda gives us ridge
set.seed(100)

#step1 build model
enetTune <- train(solTrainXtrans, solTrainY, method = "enet", tuneGrid = enetGrid, trControl = ctrl, preProc = c("center", "scale"))

# step 2 finalize model
enetTune$bestTune # reveals best model parameters

besttune <- enetTune$bestTune
fraction <- as.numeric(besttune$fraction)
lambda <- as.numeric(besttune$lambda)

enetModel <- enet(x = as.matrix(solTrainXtrans), y = solTrainY, lambda = lambda, normalize = TRUE) # normalize centers and scales the model
enetPred <- predict(enetModel, newx = as.matrix(solTestXtrans), s = fraction, mode = "fraction", type = "fit")
enetFit <- enetPred$fit

# step 3 measure effectiveness of the model

enetValues <- data.frame(obs = solTestY, pred = as.vector(enetFit))
defaultSummary(enetValues)


################################### Single layer neural net for non linear regression #############################

# 2 tuning parameters
# weight decay, a penalization method to regularize the model similar to ridge regression
# size ... i.e. how many neurons

# step 0: create tuning grid and some pre processing
tooHigh <- findCorrelation(cor(solTrainXtrans), cutoff = .75)
trainXnnet <- solTrainXtrans[, -tooHigh]
testXnnet <- solTestXtrans[, -tooHigh]

nnetGrid <- expand.grid(.decay = c(0, 0.01, .1), .size = c(1:10), .bag = c(TRUE,FALSE))

#step1 build model
nnetTune <- train(solTrainXtrans, solTrainY, method = "avNNet", tuneGrid = nnetGrid, trControl = ctrl, preProc = c("center", "scale"), MaxNWts = 10 * (ncol(trainXnnet) + 1) + 10 + 1, maxit = 500)

# step 2 finalize model
nnetTune$bestTune # reveals best model parameters
besttune <- nnetTune$bestTune
size <- besttune$size
decay <- besttune$decay

nnetModel <- avNNet(solTrainXtrans, solTrainY, size = size, decay = decay, repeats = 5, linout = TRUE, trace = FALSE, maxit = 500, MaxNWts = 10 * (ncol(solTrainXtrans) + 1) + 10 + 1)

nnetPred <- predict(nnetModel, solTestXtrans)

# step 3 measure effectiveness of the model

nnetValues <- data.frame(obs = solTestY, pred = as.vector(nnetPred))
defaultSummary(nnetValues)


################################### Multivariate Adaptive Regression Splines for non linear regression #############################

# MARS model requires very little pre-processing
# 2 tuning parameters
# degree, a tuning parameter for interaction
# nprune ... i.e. prune the number of predictors

# step 0: create tuning grid

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:50)

#step1 build model

set.seed(100)
marsTuned <- train(solTrainXtrans, solTrainY, method = "earth", tuneGrid = marsGrid, trControl = ctrl)
plot(marsTuned) # nice plot that shows RMSE v/s tuning parameters

# step 2 finalize model
marsTuned$bestTune # reveals best model parameters
besttune <- marsTuned$bestTune
nprune = besttune$nprune
degree = besttune$degree

marsPred <- predict(marsTuned, solTestXtrans)

# step 3 measure effectiveness of the model

marsValues <- data.frame(obs = solTestY, pred = as.vector(marsPred))
defaultSummary(marsValues)

# additional optional steps
varImp(marsTuned) # variable importance

################################### Support Vector Machines for non linear regression #############################

# Given a threshold set by the user (defined as Epsilon) data points with residuals within the threshold do not contribute to the regression fit
# since the squared residuals are not used, large outliers have a limited effect on the regression equation
# To estimate the model parameters, SVM uses the epsilon loss function - penalizes large residuals i.e. a cost function (C)
# regression equations in SVM can be rewritten using 'kernel' function
# linear kernels have no additional tuning parameters
# polynomial kernels have an additional degree tuning parameters
# radial basis function kernels have an additional sigma tuning parameters (Scaling)

#step1 build model

svmRTuned <- train(solTrainXtrans, solTrainY, method = "svmRadial", #other methods include "svmLinear", or "svmPoly"
				preProc = c("center", "scale"), tuneLength = 14, # The tuneLength argument will use the default grid search of 14 cost values. Sigma is is estimated analytically by default
				trControl = ctrl)

# step 2 finalize model

svmRTuned$bestTune # reveals best model parameters
besttune <- svmRTuned$bestTune
sigma <- besttune$sigma
C <- besttune$C
svmRTuned$finalModel # take a look at the best model

df <- cbind(solTrainXtrans, solTrainY)
svmRModel <- ksvm(solTrainY ~ ., data = df ,kernel ="rbfdot", kpar=list(sigma = sigma) , C=C, epsilon = 0.1)

svmRPred <- predict(svmRTuned, solTestXtrans)

# step 3 measure effectiveness of the model

svmRValues <- data.frame(obs = solTestY, pred = as.vector(svmRPred))
defaultSummary(svmRValues)


################################### K-Nearest Neighbors for non linear regression #############################

# KNN method fundamentally depends on distance (euclidean, Minkowski, etc)
# all predictors be centered and scaled prior to performing KNN
# The KNN method can have poor predictive performance when local predictor structure is not relevant to the response

# step 0 - some pre processing
knnDescr <- solTrainXtrans[, -nearZeroVar(solTrainXtrans)]

# step1 build model

knnTune <- train(knnDescr, solTrainY, method = "knn", preProc = c("center", "scale"), tuneGrid = data.frame(.k = 1:20), # up to 20 nearest neighbors
	trControl = ctrl)
	
# step 2 finalize model

knntestX <- solTestXtrans[, -nearZeroVar(solTrainXtrans)]
knnPred <- predict(knnTune, knntestX)

# step 3 measure effectiveness of the model

knnValues <- data.frame(obs = solTestY, pred = as.vector(knnPred))
defaultSummary(knnValues)


################################### Single tree for regression #############################

set.seed(100)

# step1 build model
rpartDepthTune <- train(solTrainXtrans, solTrainY, method = "rpart2", tuneLength = 10,trControl = ctrl)
# To tune over maximum depth, the method option should be set to method="rpart2"

rpartCpTune <- train(solTrainXtrans, solTrainY, method = "rpart", tuneLength = 10, trControl = ctrl)
# To tune over complexity parameter, the method option should be set to method="rpart2"

# step 2 finalize model
rpartDepthPred <- predict(rpartDepthTune, solTestXtrans)
rpartCpPred <- predict(rpartCpTune, solTestXtrans)

# step 3 measure effectiveness of the model

rpartDepthValues <- data.frame(obs = solTestY, pred = as.vector(rpartDepthPred))
defaultSummary(rpartDepthValues)

rpartCpValues <- data.frame(obs = solTestY, pred = as.vector(rpartCpPred))
defaultSummary(rpartCpValues)

# additional steps
library(partykit)
rpartTree2 <- as.party(rpartDepthTune$finalModel)
plot(rpartTree2)

################################### Bagging methods for regression #############################

# Generate a bootstrap sample of the original data --> Train an unpruned tree model on this sample
# average the predictions across models
# tuning parameter is the number of bootstrap samples to aggregate, m.



################################### Random Forests methods for regression #############################

# problem with bagging is that trees constructed from bootstrap are essentially correlated
# Random tree solves that problem by selecting a subset of predictors
# tuning parameter is the number of randomly selected predictors, k, to choose from at each split AND number of trees

# step 0: create tuning grid
ncolflr <- floor(ncol(solTrainXtrans)/4)
ncolce <- ceiling(ncol(solTrainXtrans)/1.5)
rfGrid <- expand.grid(.mtry = seq(ncolflr, ncolce, by = 10))

# step1 build model
rfTuned <- train(solTrainXtrans, solTrainY, method = "rf", ntree = 500, trControl = ctrl)

# step 2 finalize model
rfPred <- predict(rfTuned, solTestXtrans)

# step 3 measure effectiveness of the model

rfValues <- data.frame(obs = solTestY, pred = as.vector(rfPred))
defaultSummary(rfValues)

################################### Boosting methods for regression #############################

# Tuning parameters are tree depth, D, and number of iterations, K
# Compute the average response, y, and use this as the initial predicted value for each sample 
# --> for k = 1 to K do -->
		# Compute the residual
		# Fit a regression tree of depth, D, using the residuals as the response
		# Predict each sample using the regression tree fit in the previous step
		# Update the predicted value of each sample by adding the previous iteration’s predicted value to the predicted value generated in the previous step

# Boosting may lead to overfitting therefore a regularization, or shrinkage may be used.
	# only a fraction of the current predicted value is added to the previous iteration’s predicted value. 
	# This fraction is commonly referred to as the learning rate or shrinkage and is parameterized by the symbol lambda

# step 0: create tuning grid	
gbmGrid <- expand.grid(.interaction.depth = seq(1, 7, by = 2), .n.trees = seq(100, 1000, by = 50), .shrinkage = c(0.01, 0.1))

# step1 build model
set.seed(100)
gbmTune <- train(solTrainXtrans, solTrainY, method = "gbm", tuneGrid = gbmGrid, verbose = FALSE)	

# step 2 finalize model
gbmPred <- predict(gbmTune, solTestXtrans)

# step 3 measure effectiveness of the model

gbmValues <- data.frame(obs = solTestY, pred = as.vector(gbmPred))
defaultSummary(gbmValues)	

	
################################### Cubist methods for regression #############################
	
# In Cubist, the models are still combined using a linear combination of two models
# ˆypar = a × ˆy(k) + (1 − a) × ˆy(p) --> y(k) is the prediction from the current model and ˆy(p) is from parent model above it in the tree. 
# a is the mixing proportion. a = (Var[e(p)] − Cov[e(k), e(p)]) / Var[e(p)− e(k)]
# adjusted error rate is the criterion for pruning and/or combining rules
# Starting with splits made near the terminal nodes, each condition of the rules is tested using the adjusted error rate for the training set.
# If the deletion of a condition in the rule does not increase the error rate, it is dropped
# rule-based model is finalized by either using a single model or a committee. 
# Model committees can be created by generating a sequence of rule-based models
# Cubist relies on K most similar neighbors when predicting a new sample
# tuning parameters include K (neighbors) and # of committees

# Tuning parameters of cubist include numbers of committees and neighbors

# step 0: create tuning grid	
cubistGrid <- expand.grid(.committees = seq(10, 100, 10), .neighbors = seq(0, 9, 1))

# step1 build model
set.seed(100)
cubistTune <- train(solTrainXtrans, solTrainY, method = "cubist", tuneGrid = cubistGrid, trControl = ctrl)

# step 2 finalize model
cubistPred <- predict(cubistTune, solTestXtrans)

# step 3 measure effectiveness of the model

cubistValues <- data.frame(obs = solTestY, pred = as.vector(cubistPred))
defaultSummary(cubistValues)	

################################### Classification Methods #############################

# when some models are used for classification, like neural networks and partial least squares, they produce continuous predictions
# that do not follow the definition of a probability-the predicted values are not necessarily between 0 and 1 and do not sum to 1
# For classification models like these, a transformation must be used to coerced
# One such method is the softmax transformation ˆpl = e^yl / (e^y1 + e^y2 + .... + e^yC)    [l is the prob of class 'l']
# simoidal function: 1/(1+e^-z) where z is a linear function

# TP (true positive): predicted event and observed event are both positive
# FP: predicted event is positive, but observed event is negative
# FN: predicted event is negative, but observed event is positive
# TN: predicted event and observed event are both negative

# Sensitivity = number of samples with the event and predicted to have the event/ number of samples having the event i.e. TP/P (where P = number of observed positive events)
# Specificity = number of samples without the event and predicted as nonevents / number of samples without the event i.e. TN/N
# TP rate = Sensitivity
# FP rate = 1 - Specificity
# PPV (positive predicted value) = used for rare events = TP/(TP+FP)
# alternate expression for PPV = Sensitivity × Prevalence /((Sensitivity × Prevalence) + ((1 − Specif icity) × (1 − Prevalence)))

# examples:
# • Predict investment opportunities that maximize return
# • Improve customer satisfaction OR Revenue by market segmentation
# • Lower inventory costs by improving product demand forecasts or
# • Reduce costs associated with fraudulent transactions

# for this cheat sheet, we will use the caravan data.
# Caravan data is highly imbalanced. So we will use the ROSE package for dealing with imbalance and improving our accuracy
# for more information go to http://www.analyticsvidhya.com/blog/2016/03/practical-guide-deal-imbalanced-classification-problems/

library(ROSE)
library(pROC)
data <- Caravan

### There are a few predictors with only a single value, so we remove these first
isZV <- apply(data, 2, function(x) length(unique(x)) == 1)
data <- data[, !isZV]
rm(isZV)

data.x <- data[,-c(86)]
data.y <- as.data.frame(data[,c(86)])

segPP <- preProcess(data.x, method = c("YeoJohnson", "knnImpute"))
data.x <- predict(segPP, data.x)

segCorr <- cor(data.x)
highCorr <- findCorrelation(segCorr, .85)
data.x <- data.x[, -highCorr]

#nzv <- nearZeroVar(data.x)
data <- cbind(data.x,data.y)
colnames(data)[colnames(data) == 'data[, c(86)]'] <- 'Purchase'

set.seed(123)
train= sample(c(TRUE,TRUE,TRUE,FALSE), nrow(data),rep=TRUE)
test = (!train)
data.train <- data[train,]
data.test <- data[test,]

data.rose <- ROSE(Purchase ~ ., data = data.train, seed = 1)$data

ctrl <- trainControl(method = "cv", number = 10, classProbs = TRUE) 

data.roseY <- data.rose$Purchase
data.testY <- data.test$Purchase
data.roseX <- data.rose[,-c(61)]
data.testX <- data.test[,-c(61)]

################################### Logistic Regression #############################

# step1 build model
glmTune <- train(data.roseX, data.roseY, method = "glm", trControl = ctrl)

# step 2 finalize model
glm.probs=predict(glmTune,newdata=data.test,type="prob") 

# step3 evaluate model
glm.ROCR = prediction(glm.probs$Yes, data.test$Purchase)
glm.perfROCR = performance(glm.ROCR, "tpr", "fpr")
plot(glm.perfROCR, colorize=TRUE)

performance(glm.ROCR, "auc")@y.values


################################### Linear Discriminant Analysis #############################

# based on bayes rules
# assumes means of the groups are unique, covariance across predictors are identical
# optimizing function looks to maximize between group variances

# step1 build model
ldaTune <- train(data.roseX, data.roseY, method = "lda", preProc = c("center","scale"), metric = "ROC", trControl = ctrl)

# step 2 finalize model
lda.probs=predict(ldaTune,newdata=data.testX,type="prob")  # be careful not to use the full data set here

# step3 evaluate model
lda.ROCR = prediction(lda.probs$Yes, data.test$Purchase)
lda.perfROCR = performance(lda.ROCR, "tpr", "fpr")
plot(lda.perfROCR, colorize=TRUE)

performance(lda.ROCR, "auc")@y.values

################################### Partial Least Squares for classification #############################



#step 0 PLS takes numeric values for response variable, so we create a numeric response variable
data.rose$Purchase1 <- ifelse(data.rose$Purchase =="Yes",1,0)

# step1 build model
ncomp = 10
plsTune <- plsr(Purchase1 ~ . , data = data.rose[, !names(data.rose) %in% c("Purchase")], scale = T, probMethod = "Bayes", ncomp = ncomp)

# step 2 finalize model
pls.probs=predict(plsTune,newdata=data.testX,type="response") 

# step3 evaluate mode
perf <- as.vector(0)
for (i in 1:ncomp){

pls.ROCR = prediction(pls.probs[,1,i], data.test$Purchase) # here we are evaluating the model with 4 components. To evaluate with 3 components use pls.probs[,1,3]
pls.perfROCR = performance(pls.ROCR, "tpr", "fpr")
plot(pls.perfROCR, colorize=TRUE)

perf <- append(perf, as.vector(unlist(performance(pls.ROCR, "auc")@y.values)))

}

################################### Penalized Models for classification #############################

# we use the glmnet package
# If alpha=0 then a ridge regression model is fit, and if alpha=1 then a lasso model is fit
# tune for lambda. Alpha between 0 and 1 corresponds to elasticnet

# step 0: create tuning grid	
glmnGrid <- expand.grid(.alpha = c(0, .1, .2, .4, .6, .8, 1), .lambda = seq(.01, .2, length = 20))

# step1 build model
glmnTuned <- train(data.roseX, data.roseY, method = "glmnet", tuneGrid = glmnGrid, preProc = c("center", "scale"), trControl = ctrl)

# step 2 finalize model
glm.probs=predict(glmnTuned,newdata=data.testX,type="prob")  

# step3 evaluate model
glm.ROCR = prediction(glm.probs$Yes, data.test$Purchase)
glm.perfROCR = performance(glm.ROCR, "tpr", "fpr")
plot(glm.perfROCR, colorize=TRUE)

performance(glm.ROCR, "auc")@y.values