# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 18:33:11 2021

@author: MURAT
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
df = pd.read_csv("multiple_linear_regression_dataset.csv",sep=";")


x=df.iloc[:,[0,2]]
y= df.maas.values.reshape(-1,1)

multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x,y)


print("b0 : ",multiple_linear_regression.intercept_)
print("b1,b2 : ",multiple_linear_regression.coef_)


print("Deneyim ve Yaş Giriniz : ")

deneyim = int(input())
yas = int(input())

print("Tahmini Maaş : ",multiple_linear_regression.predict(np.array([[deneyim,yas]])))