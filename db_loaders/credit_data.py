# Pre processamento necessario para a base de dados de credito (credit-data.csv)
# 
from __init__ import load_db
import pandas as pd
from sklearn.preprocessing import Imputer, StandardScaler

# load_db sempre retorna um dataframe do pandas
base = load_db("credit-data")

# Podemos ver que na base há problemas com idade negativa. Vamos substituir os invalidos e os faltantes com a media (dos validos)
media_idade = base.loc[ base.age > 0 ].age.mean()

# substitui os incorretos (<= 0)
base.loc[ base.age <= 0, 'age' ] = media_idade

# Vamos agora separar nossa análise em 2 grupos: previsores e atributos classe. 
# Vamos trabalhar diretamente com as matrizes numericas do numpy
prev = base.iloc[:, 1:4].values
post = base.iloc[:, 4].values

# Valores faltantes: Podemos preencher em todas as colunas ao mesmo tempo com Imputer
imputer = Imputer(axis=0)
imputer = imputer.fit(prev[:, 0:3])
prev[:, 0:3] = imputer.transform(prev[:, 0:3])

# prev[0:5]

# Vamos padronizar a escala dos atributos, para evitar comportamentos ruins nos estimadores
# Isso torna nossas analises mais robustas contra outliers
# Este metodo usa padronizacao (standardization)
scaler = StandardScaler()
prev = scaler.fit_transform(prev)

# Agora nossos atributos previsores estao em um formato adequado para os algoritmos:
# Uma matriz do numpy, com escala padronizada.
# No caso dessa base de dados nao temos variaveis categoricas, apenas numericas de fato. 
# Nao precisaremos da etapa do labelencoder

