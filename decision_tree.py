#%%[markdown]
# # Árvores de Decisão
# O classificador por árvores de decisão funciona com base em teoria da informação.
# Primeiramente, estima-se o ganho de informacao por cada variavel preditora

#%% 
# Algoritmos de arvores de decisao para prever dados
# classificador por arvores de decisao
from sklearn.tree import DecisionTreeClassifier, export
from db_loaders.risco_credito import previsores, risco

# Classificador por arvores de decisao
modelo = DecisionTreeClassifier(criterion='entropy')
modelo.fit(previsores, risco)

export.export_graphviz(modelo, 
    out_file="assets/arvore.dot", 
    feature_names=["Histórico","Dívida","Garantias","Renda"],
    class_names=["Alto","Moderado","Baixo"],
    filled=True,
    leaves_parallel=True)

resultado = modelo.predict([[0,0,1,2],[2,0,0,0]])

#%%
