#%% [markdown] 
# # Naive Bayes com sklearn  
# Exemplos típicos do uso de Naive Bayes:
# - Filtros de spam
# - Mineração de sentimentos
# - Separação de Documentos
#
# Este algoritmo é um classificador probabilístico, baseado no *Teorema de Bayes*
# $$ P(A|B) = \frac{P(B|A)P(A)}{P(B)} $$
# O algoritmo monta uma tabela de probabilidades, e gera classificações 

#%%
import pandas as pd

base = pd.read_csv("risco_credito.csv")

#%%
