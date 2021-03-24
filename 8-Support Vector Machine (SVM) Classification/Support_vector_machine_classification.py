#!/usr/bin/env python
# coding: utf-8

# In[33]:
# Author : Murat Yıldırım

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[34]:


data = pd.read_csv("data.csv")


# In[35]:


data.drop(["id","Unnamed: 32"],axis=1,inplace=True)
data.head()


# In[36]:


M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]


# In[37]:


plt.scatter(M.radius_mean,M.texture_mean,color ="red",label="kotu",alpha=0.2)
plt.scatter(B.radius_mean,B.texture_mean,color ="green",label="iyi",alpha=0.2)
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[38]:


data.diagnosis=[1 if each == "M" else 0 for each in data.diagnosis ]
y=data.diagnosis.values
x_data=data.drop(["diagnosis"],axis=1)


# In[39]:


x= (x_data- np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[40]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state =1)


# In[52]:

from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train,y_train)



# In[53]:

print("accuracy : ",svm.score(x_test,y_test))



# In[56]:


    


# In[ ]:




