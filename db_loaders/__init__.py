# Utilidades para importar arquivos csv deste projeto
# Usaremos o pandas
import pandas as pd
''' Carrega um database para o projeto'''

def load_db(db, type="csv"):
    if type == "csv":
        return pd.read_csv("db/{0}.csv".format(db))