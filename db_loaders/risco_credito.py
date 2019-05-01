# Pre processamento de dados para a base risco-credito.csv
from db_loaders import load_db
from sklearn.preprocessing import LabelEncoder

# Carregamos a base bruta
base = load_db('risco-credito')
previsores = base.iloc[:, 0:4].values
risco = base.iloc[:, 4].values

# Quando usamos iloc ele altera tambem na base vetorial
lbl = LabelEncoder()
previsores[:,0] = lbl.fit_transform(previsores[:,0])
previsores[:,1] = lbl.fit_transform(previsores[:,1])
previsores[:,2] = lbl.fit_transform(previsores[:,2])
previsores[:,3] = lbl.fit_transform(previsores[:,3])

# Agora temos as variaveis categoricas devidamente tratadas numericamente
