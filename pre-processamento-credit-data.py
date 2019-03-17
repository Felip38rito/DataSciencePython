# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 12:29:01 2017

@author: Jones
"""
import pandas as prior
pd = prior.read_csv('credit_data.csv')
pd.describe()
pd.loc[pd['age'] < 0]
# apagar a coluna
pd.drop('age', 1, inplace=True)
# apagar somente os registros com problema
pd.drop(pd[pd.age < 0].index, inplace=True)
# preencher os valores manualmente
# preencher os valores com a média
pd.mean()
pd['age'].mean()
pd['age'][pd.age > 0].mean()
pd.loc[pd.age < 0, 'age'] = 40.92
        
prior.isnull(pd['age'])
pd.loc[prior.isnull(pd['age'])]

previsores = pd.iloc[:, 1:4].values
classe = pd.iloc[:, 4].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3])
previsores[:,0:3] = imputer.transform(previsores[:,0:3])

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
previsores = scaler.fit_transform(previsores)
                 
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  
                  