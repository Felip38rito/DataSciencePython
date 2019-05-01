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
# Carregamos as variaveis que usaremos para prever a classe
from db_loaders.risco_credito import previsores, risco
from sklearn.naive_bayes import GaussianNB, MultinomialNB

# modelo = GaussianNB()
modelo = MultinomialNB()
# Treino a partir do modelo
modelo.fit(previsores, risco)
# Predicao a partir do metodo de naive bayes
resultado = modelo.predict([[0,0,1,2], [2, 0, 0, 0]])

#%% [markdown]
# # Comentando os resultados do naive bayes:
# Quando temos dados discretos nos preditores, é adequado usar o Multinomial, e o Gaussiano quando contínuos.
# Alem disso, podemos ver que o algoritmo resolve a indicacao com tabelas de probabilidade  


#%%
