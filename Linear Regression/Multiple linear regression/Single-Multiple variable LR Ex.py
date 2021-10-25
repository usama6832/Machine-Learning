#Importing Libraries

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

#reading data

df = pd.read_csv('homeprices.csv')
df

#plotting data

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')

#Saving data into single column / dependent variable Y

df_area = df.drop('price',axis='columns')
df_area

df_price = df.drop('area',axis='columns')
df_price

# Create and training linear regression model #one variable

reg = linear_model.LinearRegression()
reg.fit(df_area,df_price)

# Create and training linear regression model #multi variable
reg = linear_model.LinearRegression()
reg.fit(df.drop('price',axis='columns'),df.price)


#Calculating coef(slop), intercept
reg.coef_
reg.intercept_

#Predicting dependent variable/price  #with single independent variable
reg.predict([[3300]])

#Predicting dependent variable/price #with multiple independent variable
reg.predict([[3000, 3, 40]])
reg.predict([[2500, 4, 5]])

#Read data from file
area_df = pd.read_csv("areas.csv")
area_df

#Price predcition for new room/ test data

price_pred = reg.predict(area_df)
price_pred

#Align predcited record with input record

area_df['prices']=p
area_df

#Save the predcited data into file name "prediction.csv"

area_df.to_csv("prediction.csv")

#Plot predicted one variable LR line 
plt.plot(area_df.area,area_df.prices,color='red',marker='+')

#Plot predicted multi variable LR line
plt.plot(area_df.area,area_df.prices,color='red',marker='+')