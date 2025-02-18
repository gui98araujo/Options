import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV

# Carregar o dataset
df_model = pd.read_csv("df_model.csv")

# Separar features e target
X = df_model.drop(columns=['[SRM] Código da operação','variavel_target','Total do Contrato (Bruto)'])
y = df_model['variavel_target']

# Normalização
geral_features = ['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score',
   'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE', 'Carater Percentual_de_divida_vencida_total',
   'Carater Quantidade_de_restricoes_comerciais', 'Serasa Quantidade de protestos', 'VTM Valor total',
   'Taxa de Juros', 'Total do Contrato (Bruto)/renda utilizada']

scaler = MinMaxScaler()
X[geral_features] = scaler.fit_transform(X[geral_features])

# Aplicar NearMiss para balanceamento
tnr = NearMiss()
X_res, y_res = tnr.fit_resample(X, y)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, stratify=y_res, random_state=0)

# Definir modelo de Decision Tree
dt_model = DecisionTreeClassifier(random_state=0)

# Definir os hiperparâmetros para otimização
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Otimizar os parâmetros usando GridSearchCV
grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_dt_model = grid_search.best_estimator_

# Fazer previsões
dt_y_pred = best_dt_model.predict(X_test)
dt_y_proba = best_dt_model.predict_proba(X_test)[:, 1]

# Exibir métricas
print("Decision Tree Classifier (Optimized):")
print(classification_report(y_test, dt_y_pred))

# Matriz de Confusão
cm = confusion_matrix(y_test, dt_y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Adimplente', 'Inadimplente'], yticklabels=['Adimplente', 'Inadimplente'])
plt.title('Decision Tree Classifier - Matriz de Confusão')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.show()

# Curva ROC
auc_score = roc_auc_score(y_test, dt_y_proba)
fpr, tpr, _ = roc_curve(y_test, dt_y_proba)
plt.figure(figsize=(10, 7))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Importância das Features
importances = best_dt_model.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 7))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='Blues_r')
plt.title('Importância das Features')
plt.show()

# Interface Streamlit
st.title('Análise de Risco de Crédito')

# Input do usuário
st.sidebar.header('Preencha os dados para simulação')
nota_clinica = st.sidebar.selectbox('Nota da Clínica', options=[0, 1, 2])
idade = st.sidebar.slider('Idade', 18, 70, 30)
endividamento = st.sidebar.slider('Endividamento (%)', 0.0, 100.0, 50.0)
serasa_score = st.sidebar.slider('Serasa Score', 300, 1000, 700)
acoes_judiciais = st.sidebar.slider('Ações Judiciais', 0, 5, 0)
percentual_divida = st.sidebar.slider('Percentual Dívida Vencida', 0.0, 100.0, 10.0)
restricoes_comerciais = st.sidebar.slider('Restrições Comerciais', 0, 5, 0)
quantidade_protestos = st.sidebar.slider('Quantidade Protestos', 0, 5, 0)
vtm_valor_total = st.sidebar.slider('VTM Valor Total', 1000.0, 50000.0, 10000.0)
taxa_juros = st.sidebar.slider('Taxa de Juros (%)', 2.0, 15.0, 5.0)
renda_contrato = st.sidebar.slider('Valor Contrato / Renda', 0.1, 10.0, 1.0)

# Criando input para modelo
dados_input = np.array([[nota_clinica, idade, endividamento, serasa_score, acoes_judiciais, percentual_divida,
                          restricoes_comerciais, quantidade_protestos, vtm_valor_total, taxa_juros, renda_contrato]])
dados_input_scaled = scaler.transform(dados_input)
