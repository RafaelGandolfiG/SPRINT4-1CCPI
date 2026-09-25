# Integrantes:
# Rafael Gandolfi Gonçalves-569036
# Rafael Lins-570588
# Cauã Paes-569906
# Guilherme Miranda-573107
# Carlos Eduardo-572949
# João Pedro Soler-569725

# Setup
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# 1a -

# carregar o csv em um DataFrame
df = pd.read_csv(r"C:\Users\Rafael\Documents\SPRINT4-1CCPI\sprint-mlpam\Renewable_Energy_Data.csv")

print("\n==========DataFrame completo==========\n")
print(df)

# 1b -

# encoding do target (Energy_Class) para uma variável numérica
df["Energy_Class"] = df["Energy_Class"].map({"Low": 0, "Medium": 1, "High": 2})

# Encoding das outras colunas categóricas
encoder = LabelEncoder()

# Region: East = 0, North = 1, South = 2, West = 3
df["Region"] = encoder.fit_transform(df["Region"])

# Energy_Source: Hydro = 0, Solar = 1, Wind = 2
df["Energy_Source"] = encoder.fit_transform(df["Energy_Source"])

# Season: Autumn = 0, Spring = 1, Summer = 2, Winter = 3
df["Season"] = encoder.fit_transform(df["Season"])

print("\n==========Tabela depois dos encodings==========\n")
print(df)

# 1c -

# matriz de correlação entre todas as colunas
correlacao = df.corr()

print("\n==========Matriz de Correlação==========\n")
print(correlacao)

# 1d -

# Com base na matriz de correlação, optamos por utilizar todas as features
# para realizar a classificação. A variável que apresentou a maior correlação
# com Energy_Class foi Efficiency_Ratio, com aproximadamente 0.21.
#
# Entretanto, esse valor ainda representa uma correlação relativamente baixa,
# enquanto as demais features apresentaram correlações ainda mais próximas de 0.
#
# Dessa forma, não existe um pequeno conjunto de features que se destaque
# claramente pela correlação com Energy_Class. Por esse motivo, utilizaremos
# todas as features no modelo de classificação, evitando descartar informações
# que, quando analisadas em conjunto, podem contribuir para a classificação.

# 2a -

# O modelo escolhido foi a Regressão Logística (LogisticRegression),
# disponível na biblioteca scikit-learn.
#
# Apesar do nome "regressão", a Regressão Logística é um modelo utilizado
# para problemas de classificação. Neste projeto, ela será utilizada para
# classificar os dados nas classes Low, Medium e High de Energy_Class.

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 2b -

# A Regressão Logística é considerada um modelo linear porque utiliza uma
# combinação linear das features para realizar a classificação.
#
# Essa combinação pode ser representada por:
# z = b0 + b1*x1 + b2*x2 + ... + bn*xn
#
# A fronteira de decisão (decision boundary) é definida a partir dessa
# combinação linear e é utilizada para separar as diferentes classes.
#
# Portanto, a característica linear do modelo está diretamente relacionada
# à forma como ele constrói sua fronteira de decisão.

# 2c -

# X recebe todas as colunas, menos a coluna que queremos prever
X = df.drop("Energy_Class", axis=1)

# y recebe apenas a coluna que queremos prever
y = df["Energy_Class"]

print("\n==========Features utilizadas (X)==========\n")
print(X)

print("\n==========Target (y)==========\n")
print(y)

# 2d -

# Cenário 1 -

# 60% dos dados serão utilizados para treinamento
# e 40% serão utilizados para teste.
X_treino_40, X_teste_40, y_treino_40, y_teste_40 = train_test_split(X, y, test_size=0.40, random_state=42)

print("\n==========Cenário 1 - 40% para teste==========\n")

print("Quantidade de dados para treinamento:", len(X_treino_40))
print("Quantidade de dados para teste:", len(X_teste_40))

# Padronização das features
scaler_40 = StandardScaler()

# Aprende a média e o desvio padrão dos dados de treinamento
# e realiza a padronização
X_treino_40 = scaler_40.fit_transform(X_treino_40)

# Padroniza os dados de teste utilizando os valores
# aprendidos nos dados de treinamento
X_teste_40 = scaler_40.transform(X_teste_40)

# Criação do modelo
modelo_40 = LogisticRegression(max_iter=1000)

# Treinamento do modelo
modelo_40.fit(X_treino_40, y_treino_40)

# Realização das previsões
y_pred_40 = modelo_40.predict(X_teste_40)

print("\nValores reais:")
print(y_teste_40.to_numpy())

print("\nValores previstos:")
print(y_pred_40)

# Cenário 2 -

# 85% dos dados serão utilizados para treinamento
# e 15% serão utilizados para teste.
X_treino_15, X_teste_15, y_treino_15, y_teste_15 = train_test_split(X, y, test_size=0.15, random_state=42)

print("\n==========Cenário 2 - 15% para teste==========\n")

print("Quantidade de dados para treinamento:", len(X_treino_15))
print("Quantidade de dados para teste:", len(X_teste_15))

# Padronização das features
scaler_15 = StandardScaler()

# Aprende a média e o desvio padrão dos dados de treinamento
# e realiza a padronização
X_treino_15 = scaler_15.fit_transform(X_treino_15)

# Padroniza os dados de teste utilizando os valores
# aprendidos nos dados de treinamento
X_teste_15 = scaler_15.transform(X_teste_15)

# Criação do modelo
modelo_15 = LogisticRegression(max_iter=1000)

# Treinamento do modelo
modelo_15.fit(X_treino_15, y_treino_15)

# Realização das previsões
y_pred_15 = modelo_15.predict(X_teste_15)

print("\nValores reais:")
print(y_teste_15.to_numpy())

print("\nValores previstos:")
print(y_pred_15)

# 3 -
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,)

# 3a -

# A performance do modelo pode ser considerada satisfatória nos dois cenários.
#
# No cenário com 40% dos dados para teste, o modelo apresentou acurácia
# de 73%, precisão de aproximadamente 74.19% e recall de 73%.
#
# No cenário com 15% dos dados para teste, o modelo apresentou acurácia
# de aproximadamente 74.67%, precisão de aproximadamente 75.51%
# e recall de aproximadamente 74.67%.
#
# Portanto, o modelo conseguiu classificar corretamente a maioria dos
# dados nos dois cenários, apresentando resultados ligeiramente melhores
# quando foram utilizados 15% dos dados para teste.

# 3b -

# Cenário 1 -

matriz_40 = confusion_matrix(y_teste_40, y_pred_40)

print("\n==========Matriz de Confusão - 40% para teste==========\n")
print(matriz_40)

# Cenário 2 -

matriz_15 = confusion_matrix(y_teste_15, y_pred_15)

print("\n==========Matriz de Confusão - 15% para teste==========\n")
print(matriz_15)

# 3c -

# Cenário 1 -

acuracia_40 = accuracy_score(y_teste_40, y_pred_40)
precisao_40 = precision_score(y_teste_40, y_pred_40, average="weighted")
recall_40 = recall_score(y_teste_40, y_pred_40, average="weighted")

print("\n==========Métricas - 40% para teste==========\n")
print("Acurácia:", acuracia_40)
print("Precisão:", precisao_40)
print("Recall:", recall_40)

# Cenário 2 -

acuracia_15 = accuracy_score(y_teste_15, y_pred_15)
precisao_15 = precision_score(y_teste_15, y_pred_15, average="weighted")
recall_15 = recall_score(y_teste_15, y_pred_15, average="weighted")

print("\n==========Métricas - 15% para teste==========\n")
print("Acurácia:", acuracia_15)
print("Precisão:", precisao_15)
print("Recall:", recall_15)

# 3d -

# Com 40% dos dados para teste, o modelo obteve:
# Acurácia: 73%
# Precisão: aproximadamente 74.19%
# Recall: 73%
#
# Com 15% dos dados para teste, o modelo obteve:
# Acurácia: aproximadamente 74.67%
# Precisão: aproximadamente 75.51%
# Recall: aproximadamente 74.67%.
#
# A partir desses resultados, podemos observar que o modelo apresentou
# uma performance satisfatória nos dois cenários.
#
# O cenário com 15% para teste apresentou resultados um pouco melhores
# nas três métricas avaliadas.
#
# Nas matrizes de confusão, os valores da diagonal principal representam
# as classificações corretas e os valores fora da diagonal representam
# as classificações incorretas.

# 3e -

# Comparando os dois cenários, o conjunto com 15% para teste apresentou
# resultados ligeiramente melhores que o conjunto com 40% para teste.
#
# A acurácia aumentou de 73% para aproximadamente 74.67%.
# A precisão aumentou de aproximadamente 74.19% para 75.51%.
# O recall aumentou de 73% para aproximadamente 74.67%.
#
# Com 40% para teste, temos uma quantidade maior de dados para avaliar
# o modelo, o que permite realizar a avaliação sobre uma amostra maior.
# Entretanto, apenas 60% dos dados ficam disponíveis para treinamento.
#
# Com 15% para teste, 85% dos dados ficam disponíveis para treinamento,
# permitindo que o modelo aprenda utilizando uma quantidade maior de dados.
# Por outro lado, a avaliação é realizada utilizando uma amostra menor.
#
# Neste dataset e nesta divisão específica, utilizar 15% para teste
# apresentou um desempenho ligeiramente superior.
