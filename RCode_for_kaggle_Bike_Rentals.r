'
datetime - hourly date + timestamp  
season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy 
	2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 
	3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 
	4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals
'
# load the libraries

library(psych)
library(MASS)
library(caret)
library(pls)
library(gam)
library(splines)
library(earth) #MARS models
library(gbm)

# first some data wrangling
data.original.train <- read.csv("Reg_train.csv")
data.tr <- data.original.train 
data.tr$datetime <- as.POSIXct(data.tr$datetime, tz = "")
data.tr$time <- format(as.POSIXct(data.tr$datetime, format="%Y-%m-%d %H:%M"), format="%H:%M")
data.tr$month <- format(as.POSIXct(data.tr$datetime, format="%Y-%m-%d %H:%M"), format="%m")
data.tr$time <- as.factor(data.tr$time)
data.tr$month <- as.factor(data.tr$month)
data.tr$season <- as.factor(data.tr$season)
data.tr$holiday <- as.factor(data.tr$holiday)
data.tr$workingday <- as.factor(data.tr$workingday)
data.tr$weather <- as.factor(data.tr$weather)
data.tr$count <- NULL
# we separate the y variables in 2. One Y variable is casual and the other is registered. We will evaluate models that predict these 2 variables and add them for final numbers
data.tr.cas <- data.tr 
data.tr.reg <- data.tr


# data wrangling of data.tr.reg data frame
data.tr.reg$casual <- NULL
data.tr.reg$registered <- (data.tr.reg$registered)^(1/3) ### normalizes y variable
data.tr.reg$holiday <- NULL
data.tr.reg$season <- NULL
data.tr.reg$temp <- NULL

# data wrangling of data.tr.cas data frame
data.tr.cas$registered <- NULL
data.tr.cas$casual <- (data.tr.cas$casual)^(1/3) ### normalizes y variable

'
# upon running boost on the data, it is inferred that holiday, season and temp dont play important roles in the model. Therefore simplified the model.
# trying linear and Robust linear model (MASS package)
temp <- data.tr.cas
temp$datetime <- as.numeric(temp$datetime)
gbmMod1 <- gbm(casual ~ ., data = temp, distribution= "gaussian", n.trees =500 , shrinkage =0.01, interaction.depth = 1)
'

ctrl <- trainControl(method = "cv", number = 10)

lmMod2 <- train(registered ~ ., data = data.tr.reg, method = "lm", trControl = ctrl)
summary(lmMod2) # R2 = 0.8082

rlmMod1 <- rlm(registered ~ ., data = data.tr.reg)
summary(rlmMod1)
SSE = sum((rlmMod1$fitted.values - data.tr.reg$registered)^2)
SST = sum((mean(data.tr.reg$registered) - data.tr.reg$registered)^2)
R2 = 1 - (SSE/SST) #R2 = 0.799

rlmMod2 <- train(registered ~ ., data = data.tr.reg, method = "rlm", preProcess = "pca", trControl = ctrl)
summary(rlmMod2)
rlmMod <- rlmMod2$finalModel
SSE = sum((rlmMod$fitted.values - data.tr.reg$registered)^2)
SST = sum((mean(data.tr.reg$registered) - data.tr.reg$registered)^2)
R2 = 1 - (SSE/SST) #R2 = 0.71

# partial least squares

plsMod1 <- plsr(registered ~ ., data = data.tr.reg, validation = "CV")
summary(plsMod1)

plsMod2 <- train(registered ~ ., data = data.tr.reg, method = "pls", tuneLength = 42, trControl = ctrl)

# general additive model

# since time is one of the parameters, we first we evaluate how many degrees of freedom for smoothing spline
attach(data.tr.reg)
fit <- smooth.spline(datetime , registered , cv=T) # calling fit reveals ideal number of df

plot(datetime , registered , cex =.5, col = "darkgrey")
lines(fit ,col ="red",lwd =2) 

#gamMod1 <- gam(registered ~ s(datetime,15)+ season + holiday + workingday + weather + temp + atemp + humidity + windspeed + time + month, data = data.tr.reg)
gamMod1 <- gam(registered ~ s(datetime,15) + weather + workingday + atemp + humidity + windspeed + time + month, data = data.tr.reg)

SSE = sum((gamMod1$fitted.values - data.tr.reg$registered)^2)
SST = sum((mean(data.tr.reg$registered) - data.tr.reg$registered)^2)
R2 = 1 - (SSE/SST) # R2 = 0.812

gamMod2 <- train(registered ~ s(datetime,15) + weather + workingday + atemp + humidity + windspeed + time + month, data = data.tr.reg, method = "gam", df=15, trControl = ctrl)

# svm

svmMod1 <- train(registered ~ ., data = data.tr.reg, method = "svmLinear", preProc = c("center", "scale"), tuneLength = 14, trControl = ctrl) #long computing time

# Multivariate Adaptive Regression Splines

marMod1 <- earth(registered ~ ., data = data.tr.reg)
summary(marMod1)

marsGrid <- expand.grid(.degree = 1:2, .nprune = 2:32)
marMod2 <- train(registered ~ ., data = data.tr.reg, method = "earth", tuneGrid = marsGrid, trControl = ctrl)
marMod <- finalModel # very nice R-square of 0.897. 
evimp(marMod)
#expand terms to 50 tomorrow

