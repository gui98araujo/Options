import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV
import catboost 
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

# Interface Streamlit
st.title('Análise de Risco de Crédito')

# Escolha do modelo
model_choice = st.sidebar.selectbox("Escolha o modelo", ["Decision Tree", "Neural Network"])

from catboost import CatBoostClassifier

if model_choice == "CatBoost":
    model = CatBoostClassifier(verbose=0, random_seed=0)
    param_grid = {
        'iterations': [100, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1],
        'depth': [4, 6, 10],
        'l2_leaf_reg': [1, 3, 5]
    }
elif model_choice == "Neural Network":
    model = MLPClassifier(max_iter=500, random_state=0)
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001]
    }

# Otimizar os parâmetros usando GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_model = grid_search.best_estimator_

# Fazer previsões
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]
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

# Botão de Simulação
if st.sidebar.button("Simular"):
    resultado = best_model.predict(dados_input_scaled)
    probabilidade = best_model.predict_proba(dados_input_scaled)[:, 1]
    
    st.subheader("Resultado da Simulação")
    st.write(f"Probabilidade de inadimplência: {probabilidade[0]:.2f}")
    st.write("Status Previsto:", "Inadimplente" if resultado[0] == 1 else "Adimplente")
    
    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Adimplente', 'Inadimplente'], yticklabels=['Adimplente', 'Inadimplente'])
    ax.set_title('Matriz de Confusão')
    ax.set_xlabel('Predito')
    ax.set_ylabel('Real')
    st.pyplot(fig)
    
    # Curva ROC
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
    ax.plot([0, 1], [0, 1], color='red', linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Curva ROC')
    ax.legend(loc='lower right')
    st.pyplot(fig)
