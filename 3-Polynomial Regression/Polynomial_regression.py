# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:59:32 2021

@author: MURAT
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures





df = pd.read_csv("polynomial regression.csv",sep=";")

y=df.araba_max_hiz.values.reshape(-1,1)

x=df.araba_fiyat.values.reshape(-1,1)

polynomial_regression = PolynomialFeatures(degree=2)

x_polynomial = polynomial_regression.fit_transform(x)


linear_regression = LinearRegression()

linear_regression.fit(x_polynomial,y)

y_head =linear_regression.predict(x_polynomial)

plt.plot(x,y_head,color="red",label="poly")

plt.show()