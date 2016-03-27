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

