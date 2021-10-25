#Import libraries

import pandas as pd
#Create dataframe

df = pd.read_csv("salaries.csv")
df.head()

#Drop data columns
#Seperate Input Output

inputs = df.drop('salary_more_then_100k',axis='columns')

target = df['salary_more_then_100k']

#Create label encoder/object for text data

from sklearn.preprocessing import LabelEncoder
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()

#Transform the text data into numerical form/label 

inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])

#show the data
inputs

#Drop text data column from the data table

inputs_n = inputs.drop(['company','job','degree'],axis='columns')

#show the data-input and target
inputs_n

target

#Decision Tree algorithm and training over data 
from sklearn import tree
model = tree.DecisionTreeClassifier()

model.fit(inputs_n, target)

#Score of the model
model.score(inputs_n,target)

#Predicting the salary of employe

Is salary of Google, Computer Engineer, Bachelors degree > 100 k ?
model.predict([[2,1,0]])

Is salary of Google, Computer Engineer, Masters degree > 100 k ?

model.predict([[2,1,1]])

