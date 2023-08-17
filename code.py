import numpy as np

Import matplotlib.pyplot as plt

import pandas as pd

from sklearn import metrics

from sklearn import preprocessing from sklearn.model selection Import train_test_split from sklearn.linear model import LinearRegression

data.head()

data pd.read_csv("C:\\Users\\del\\Desktop\\project\\TSLA.csv") data.describe()

data.Info()

X-data[["High". "Low", "Open", "Volume"]].values

y data['close').values

print(X)

print(y)

Split dota into testing and training sets

X_train, X_test, y train, y test train_test_split(x,y, test size-8.3, random_state-1)

afrom sklearn. Linear model import LinearRegression

Create Regression Model Model LinearRegression()

Train the model

Model.fit(x train, y_train) Printing Coefficient print(Model.coef)

Use model to make predictions

predicted Model.predict(x_test) print(predicted)

datal - pd.DataFrame({"Actual": y_test. flatten(), 'Predicted: predicted. Flatten()}) datal.head(20)

import math

print("Mean Absolute Error:', metrics.mean absolute_error(y_test,predicted))

print("Mean Squared Error:", metrics.mean squared errorty test,predicted)) print("Root Mean Squared Error:', math.sqrt(metrics.mean squared error(y_test,predicted)))

graph datal.head(20) graph.plot(kind-"bar")
