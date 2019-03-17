#%% [markdown]
# # Curso Machine Learning com python
# ### Limpeza dos dados da base (teste) de crédito

# Primeiro importamos os dados

#%% 
import pandas as pd
bcred = pd.read_csv("credit-data.csv")
bcred.describe()

#%% [markdown]
# Percebemos que há valores inválidos em age. 
# Vamos preencher valores faltantes com a media e corrigir valores negativos

#%% 
# Calculo a media de idade dos validos
media = bcred['age'][bcred.age > 0].mean()
# substituo os incorretos
bcred.loc[ bcred.age < 0, 'age' ] = media


