# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 18:45:24 2020
@author: ariel

Implenting Simple Linear Regression from scratch.

background:
    Problem type: regression
    
    assumptions on data: this method assumes linear relationship between the dependent (y)
    and independent (x) variables.
    there is also ONE variable of each, and they follow the equation: y=ax+b where a,b are constants.
    
    Algorithm: the goal of the algorithm is to create the trend line that is closest to all of the
    data points. the distance from all data points is our Loss Function and is calculated by the
    Mean of Squared Errors: MSE (error also called residuals). the errors are squared so we can handle
    positive numbers, while emphasising the error rate for each prediction.
    by minimizing the loss function, we get the best trend line to predict next values.
    this is done by the derivitive of our Loss function:
    proof: https://youtu.be/mIx2Oj5y9Q8
    
    
    later we will calculate the covariance, which states how much do two variables change together.
    those will allow us to calculate the coefficients a,b the will determine the trend line!
    the slope (a) is calculated by dividing the covariance by the variance:
    proof: https://youtu.be/ualmyZiPs9w
    the intercept (b)
    proof: https://youtu.be/8RSTQl0bQuw
    
    Assesing algorithm's performance: using RMSE (root of Mean of Squared Errors), which gives
    the standard deviation of the residuals (prediction errors). it tells us how concentrated
    the data is around our trend line of predictions.
    we will also use r-squared, which represents the proportion of the variance
    for a dependent variable that's explained by an independent variable or variables
    in our regression model. https://youtu.be/Fc5t_5r_7IU
    
    
"""

""""plotting data to prove thee linear assumption"""
"""importing the dataset"""
import matplotlib.pyplot as plt
import pandas as pd
dataset = pd.read_csv('Salary_Data.csv')
""""x, y split"""
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)
""""plotting the data"""
plt.scatter(x, y, color = 'red')
plt.title('Salary VS Experience')
plt.xlabel('Years of experince')
plt.ylabel('Salary')
plt.show

"""data seems to fit the linear regression assumptions. now we'll create the model"""
"""calculate MSE"""
def Mean(values):
    return sum(values) / len(values)

def SumOfSquaredErrors(values, mean):
    return sum([(value - mean)**2 for value in values])

def Variance(x):
    meanX = Mean(x)
    return SumOfSquaredErrors(x, meanX)

"""Calculate covariance between x and y"""
def Covariance(x, meanX, y, meanY):
	covar = 0.0
	for i in range(len(x)):
		covar += ((x[i] - meanX) * (y[i] - meanY))
	return covar

"""calculate coefficients"""
def Coefficients(covar, varianceX, meanX, meanY):
    a = covar / varianceX #slope
    b = meanY - (a * meanX)
    return [a, b]

def TrendLinePredict(a, b, x):
    y = a*x + b
    return y

      

meanX = Mean(xTrain)
meanY = Mean(yTrain)
covar = Covariance(xTrain, meanX, yTrain, meanY)
varianceX = Variance(xTrain)
coefficients = Coefficients(covar, varianceX, meanX, meanY)
trendLine = TrendLinePredict(coefficients[0], coefficients[1], xTrain)

plt.scatter(xTrain, yTrain, color = 'red')
plt.plot(xTrain, trendLine, color = 'blue')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of experince')
plt.ylabel('Salary')
plt.show

plt.scatter(xTest, yTest, color = 'red')
plt.plot(xTrain, trendLine, color = 'blue')
plt.title('Salary VS Experience (training set)')
plt.xlabel('Years of experince')
plt.ylabel('Salary')
plt.show

def TestPredictions(xTest, coefficients):
    yPred = []
    for i in xTest:
        yPred.append(i*coefficients[0] + coefficients[1])
    return yPred

import math
def RMSE(yPred, yTest):
    return math.sqrt((sum((yPred - yTest)**2))/ len(yTest))

yPred = TestPredictions(xTest, coefficients)
print(RMSE(yPred, yTest))

def Rsquared(yTest, yPred, meanY):
    squaredErrorFromLine = (yPred - yTest)**2
    squaredErrorFromMeanY = (yPred - meanY)**2
    sumSquaredFromLine = sum(squaredErrorFromLine)
    sumsquaredFromMeanY = sum(squaredErrorFromMeanY)
    return 1 - (sumSquaredFromLine / sumsquaredFromMeanY)

print(Rsquared(yTest, yPred, meanY))















