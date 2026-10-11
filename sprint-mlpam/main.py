# 2º SEMESTRE - CHALLENGE SPRINT 4
# Integrantes:
# Rafael Gandolfi Gonçalves - RM 569036
# Rafael Lins - RM 570588
# Cauã Paes - RM 569906
# Guilherme Miranda - RM 573107
# Carlos Eduardo - RM 572949
# João Pedro Soler - RM 569725

# Bibliotecas utilizadas nas aulas e complementos necessários ao enunciado.
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

# 1a - Carregar o CSV em um DataFrame.
df = pd.read_csv("Renewable_Energy_Data.csv")
print("\n========== DATAFRAME ORIGINAL ==========")
print(df)

# 1b - Encoding do target: Low=0, Medium=1, High=2.
df["Energy_Class"] = df["Energy_Class"].map({"Low": 0, "Medium": 1, "High": 2})

# Complemento necessário: transformar features categóricas em números.
# LabelEncoder atribui códigos numéricos às categorias.
encoder = LabelEncoder()
df["Region"] = encoder.fit_transform(df["Region"])
df["Energy_Source"] = encoder.fit_transform(df["Energy_Source"])
df["Season"] = encoder.fit_transform(df["Season"])
print("\n========== DATAFRAME CODIFICADO ==========")
print(df)

# 1c - Matriz de correlação linear de Pearson (Aula 07).
correlacao = df.corr()
print("\n========== MATRIZ DE CORRELAÇÃO ==========")
print(correlacao)
print("\n========== CORRELAÇÃO COM ENERGY_CLASS ==========")
print(correlacao["Energy_Class"])

# 1d - Escolha das features.
# A matriz de correlação mostra relações lineares individuais entre as
# variáveis. Como não há um conjunto pequeno que se destaque claramente,
# optamos por manter todas as features. Uma correlação baixa isolada não
# significa necessariamente que uma variável seja inútil quando combinada
# com outras. Os códigos atribuídos às categorias não representam uma
# ordem quantitativa real, então suas correlações devem ser interpretadas
# com cautela.

# 2a - Modelo escolhido: Regressão Logística (Aula 08).
# É um classificador que pode ser utilizado com múltiplas classes.

# 2b - Por que o modelo é linear?
# O modelo calcula combinações lineares das features:
# z = b0 + b1*x1 + b2*x2 + ... + bn*xn.
# As fronteiras de decisão são lineares no espaço das features utilizadas.
# Apesar do nome, Regressão Logística é usada para classificação.

# 2c - Separação das features (X) e do target (y).
X = df.drop("Energy_Class", axis=1)
y = df["Energy_Class"]
print("\n========== FEATURES (X) ==========")
print(X)
print("\n========== TARGET (y) ==========")
print(y)

# 2d - Cenário 1: 40% para teste, 60% para treinamento.
# train_test_split é um complemento necessário ao enunciado.
X_treino_40, X_teste_40, y_treino_40, y_teste_40 = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)
# StandardScaler padroniza as variáveis; ajustamos apenas com o treino.
scaler_40 = StandardScaler()
X_treino_40 = scaler_40.fit_transform(X_treino_40)
X_teste_40 = scaler_40.transform(X_teste_40)
modelo_40 = LogisticRegression(max_iter=1000)
modelo_40.fit(X_treino_40, y_treino_40)
y_pred_40 = modelo_40.predict(X_teste_40)
print("\n========== CENÁRIO 1 - 40% PARA TESTE ==========")
print("Treinamento:", len(X_treino_40), "| Teste:", len(X_teste_40))
print("Valores reais:", y_teste_40.to_numpy())
print("Valores previstos:", y_pred_40)

# 2d - Cenário 2: 15% para teste, 85% para treinamento.
X_treino_15, X_teste_15, y_treino_15, y_teste_15 = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
scaler_15 = StandardScaler()
X_treino_15 = scaler_15.fit_transform(X_treino_15)
X_teste_15 = scaler_15.transform(X_teste_15)
modelo_15 = LogisticRegression(max_iter=1000)
modelo_15.fit(X_treino_15, y_treino_15)
y_pred_15 = modelo_15.predict(X_teste_15)
print("\n========== CENÁRIO 2 - 15% PARA TESTE ==========")
print("Treinamento:", len(X_treino_15), "| Teste:", len(X_teste_15))
print("Valores reais:", y_teste_15.to_numpy())
print("Valores previstos:", y_pred_15)


# 3a -
# A performance do modelo pode ser considerada satisfatória
# nos dois cenários, pois a acurácia ficou próxima de 80%.
#
# Entretanto, a acurácia sozinha não é suficiente para avaliar
# completamente o modelo. Por isso, também utilizamos
# a matriz de confusão, a precisão e o recall.

# 3b - Matrizes de confusão (Aula 08).
matriz_40 = confusion_matrix(y_teste_40, y_pred_40, labels=[0, 1, 2])
matriz_15 = confusion_matrix(y_teste_15, y_pred_15, labels=[0, 1, 2])
print("\n========== MATRIZ DE CONFUSÃO - 40% ==========")
print(matriz_40)
print("\n========== MATRIZ DE CONFUSÃO - 15% ==========")
print(matriz_15)

# 3c - Acurácia (Aula 08), precisão e recall (complementos necessários).
# average='weighted' considera a quantidade de exemplos em cada classe.
acuracia_40 = accuracy_score(y_teste_40, y_pred_40)
precisao_40 = precision_score(
    y_teste_40, y_pred_40, average="weighted", zero_division=0
)
recall_40 = recall_score(y_teste_40, y_pred_40, average="weighted", zero_division=0)
acuracia_15 = accuracy_score(y_teste_15, y_pred_15)
precisao_15 = precision_score(
    y_teste_15, y_pred_15, average="weighted", zero_division=0
)
recall_15 = recall_score(y_teste_15, y_pred_15, average="weighted", zero_division=0)
print("\n========== MÉTRICAS - 40% ==========")
print(f"Acurácia: {acuracia_40:.2%}")
print(f"Precisão: {precisao_40:.2%}")
print(f"Recall: {recall_40:.2%}")
print("\n========== MÉTRICAS - 15% ==========")
print(f"Acurácia: {acuracia_15:.2%}")
print(f"Precisão: {precisao_15:.2%}")
print(f"Recall: {recall_15:.2%}")

# 3d -
# A matriz de confusão permite identificar os acertos e erros
# de classificação em cada uma das três classes.
#
# A acurácia representa a proporção de classificações corretas.
# A precisão avalia a proporção de previsões corretas entre
# as previsões realizadas para cada classe.
# O recall avalia a capacidade de identificar os exemplos
# que realmente pertencem a cada classe.
#
# As métricas permitem comparar o desempenho nos dois cenários.

print("\n========== ANÁLISE DOS RESULTADOS ==========")

print(f"Acurácia com 40% para teste: {acuracia_40:.2%}")
print(f"Acurácia com 15% para teste: {acuracia_15:.2%}")

print(f"Precisão com 40% para teste: {precisao_40:.2%}")
print(f"Precisão com 15% para teste: {precisao_15:.2%}")

print(f"Recall com 40% para teste: {recall_40:.2%}")
print(f"Recall com 15% para teste: {recall_15:.2%}")

# 3e - Comparação dos dois cenários.
# Com 40% para teste, avaliamos o modelo com mais registros, mas treinamos
# com apenas 60%. Com 15% para teste, treinamos com 85%, porém a avaliação
# utiliza menos registros e pode variar mais entre diferentes divisões.
# Uma diferença pequena não garante que uma divisão seja sempre melhor.
print("\n========== COMPARAÇÃO ENTRE OS CENÁRIOS ==========")
print(
    f"Diferença de acurácia (15% - 40%): {(acuracia_15 - acuracia_40) * 100:.2f} pontos percentuais"
)
if acuracia_15 > acuracia_40:
    print("Nesta divisão, o cenário com 15% para teste teve maior acurácia.")
elif acuracia_15 < acuracia_40:
    print("Nesta divisão, o cenário com 40% para teste teve maior acurácia.")
else:
    print("Nesta divisão, os dois cenários tiveram a mesma acurácia.")
