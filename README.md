# Cement_Regression
# Concrete Strength Regression Models

## Overview
We will investigate the strength of concrete and the relationships of age, water components, etc. and create regression models to predict the strength of concrete. 

### Data
The data is imported from Kaggle and is originally sourced from Prof. I-Cheng Yeh at Chung-Hua University. Here is a link to the original source: https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength

### Checking and Cleaning the Data
import pandas as pd

df = pd.read_csv("C:/Users/kashi/OneDrive/Desktop/concrete/Concrete_Data_Yeh.csv")
df.head()
df.describe()
df.isna().sum()
There are no null values in any of the columns or any negative values. We can proceed to split the data for training and testing.

### Training/Testing Data
We drop the csMPa column which is the variable that we are observing as a result of independent variables. Concrete compressive strength is csMPa.
x = df.drop('csMPa', axis = 1)
x
y = df.csMPa
y
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
### Load Libraries for Linear Model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score


model = linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred_train = model.predict(x_train)


print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_train, y_pred_train))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_train, y_pred_train))
     
From the linear model, we find that the MSE is very high and the coefficient of determination is not close to 1. Therefore, a linear model is not a good fit for the data. We find this in the testing data as well.
y_pred_test = model.predict(x_test)
     

print('Coefficients:', model.coef_)
print('Intercept:', model.intercept_)
print('Mean squared error (MSE): %.2f'
      % mean_squared_error(y_test, y_pred_test))
print('Coefficient of determination (R^2): %.2f'
      % r2_score(y_test, y_pred_test))
### Lineaer Model Plot
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(5,11))

# 2 row, 1 column, plot 1
plt.subplot(2, 1, 1)
plt.scatter(x=y_train, y=y_pred_train, c="#7CAE00", alpha=0.3)

# Add trendline
# https://stackoverflow.com/questions/26447191/how-to-add-trendline-in-python-matplotlib-dot-scatter-graphs
z = np.polyfit(y_train, y_pred_train, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"#F8766D")

plt.ylabel('Predicted csMPa')


# 2 row, 1 column, plot 2
plt.subplot(2, 1, 2)
plt.scatter(x=y_test, y=y_pred_test, c="#619CFF", alpha=0.3)

z = np.polyfit(y_test, y_pred_test, 1)
p = np.poly1d(z)
plt.plot(y_test,p(y_test),"#F8766D")

plt.ylabel('Predicted csMPa')
plt.xlabel('Experimental csMPa')

plt.savefig('plot_vertical_logS.png')
plt.savefig('plot_vertical_logS.pdf')
plt.show()
From the plots above we see that the linear regression model is not a good fit for the data.

### Install Libraries for Random Forest
!pip install tensorflow

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import seaborn as sns
pip install numpy==1.21 --user
from sklearn.ensemble import RandomForestRegressor
### Correlation Heat Map
We see below that cement, age, and fly ash have the highest correlations with the strength of the cement.
sns.heatmap(df.corr(), annot=True, cmap='Blues')
### Random Forest Model
regressor = RandomForestRegressor(n_estimators = 300, random_state = 12)
regressor.fit(x_train, y_train)
pred = regressor.predict(x_test)

plt.scatter(y_test, pred, c='b', s=0.5)
plt.xlabel('Experiment')
plt.ylabel('Prediction')
mse = mean_squared_error(y_test, pred)
rmse = mse**.5
print(mse)
print(rmse)
The Mean-Squared Error for the random forest is much lower than the linear model. The linear model had a MSE of 115 while the random forest only has 23.4. Therefore we can conclude that the random forest is a better model to predict the strength of the cement than a linear model.
