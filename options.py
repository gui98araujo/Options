import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from imblearn.under_sampling import NearMiss

# Função para transformar a nota
def transformar_nota(nota):
    if nota in [0, 1, 2, 3]:
        return 0
    elif nota in [4, 5, 6, 7]:
        return 1
    else:
        return 2

# Criando a interface Streamlit
st.title("Análise de Risco de Crédito")

menu = ["Rede Neural", "Decision Tree"]
escolha = st.sidebar.selectbox("Escolha o Modelo", menu)

# Inputs do usuário
nota_clinica = transformar_nota(st.number_input("Nota da Clínica (1-10)", min_value=1, max_value=10, step=1))
idade = st.number_input("Idade", min_value=18, max_value=100, step=1)
endividamento = st.number_input("Endividamento", min_value=0.0, max_value=1.0, step=0.01)
serasa_score = st.number_input("Serasa Score", min_value=0, max_value=1000, step=1)
acoes_judiciais = st.number_input("Ações Judiciais", min_value=0, step=1)
divida_vencida = st.number_input("Percentual de Dívida Vencida", min_value=0.0, max_value=1.0, step=0.01)
restricoes_comerciais = st.number_input("Restrições Comerciais", min_value=0, step=1)
protestos = st.number_input("Quantidade de Protestos", min_value=0, step=1)
vtm_valor = st.number_input("VTM Valor Total", min_value=0.0, step=0.01)
taxa_juros = st.number_input("Taxa de Juros", min_value=0.0, max_value=1.0, step=0.01)
valor_contrato = st.number_input("Valor do Contrato (Bruto)", min_value=0.01, step=0.01)
renda_solicitante = st.number_input("Renda do Solicitante", min_value=0.01, step=0.01)

# Criando a variável derivada
total_contrato_renda = valor_contrato / renda_solicitante

# Criando o DataFrame de entrada
input_data = pd.DataFrame([[nota_clinica, idade, endividamento, serasa_score, acoes_judiciais,
                            divida_vencida, restricoes_comerciais, protestos, vtm_valor,
                            taxa_juros, total_contrato_renda]],
                          columns=['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score',
                                   'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE',
                                   'Carater Percentual_de_divida_vencida_total',
                                   'Carater Quantidade_de_restricoes_comerciais',
                                   'Serasa Quantidade de protestos', 'VTM Valor total', 'Taxa de Juros',
                                   'Total do Contrato (Bruto)/renda utilizada'])

# Gerar os dados fictícios (substituir pelo dataset real)
df = pd.read_csv("df_model.csv")  # Suponha que este seja o dataset real
X = df.drop(columns=['[SRM] Código da operação','variavel_target','Total do Contrato (Bruto)'])
y = df['variavel_target']

# Normalização
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
input_data = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)

# Balanceamento com NearMiss
nr = NearMiss()
X_res, y_res = nr.fit_resample(X, y)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, stratify=y_res, random_state=0)

# Treinamento do modelo Decision Tree
modelo = DecisionTreeClassifier()
modelo.fit(X_train, y_train)
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

# Predição do caso inputado
pred_input = modelo.predict(input_data)[0]
pred_input_prob = modelo.predict_proba(input_data)[0][1]
cor_predicao = 'red' if pred_input_prob > 0.5 else 'green'

# Exibição dos resultados
if st.button("Simular"):
    st.markdown(f"### Probabilidade de Inadimplência: <span style='color:{cor_predicao}'> {pred_input_prob:.2f} </span>", unsafe_allow_html=True)
    
    # Exibir métricas
    report = classification_report(y_test, y_pred, output_dict=True)
    df_metrics = pd.DataFrame(report).T
    st.dataframe(df_metrics.style.applymap(lambda x: "background-color: lightcoral" if x < 0.6 else "background-color: lightgreen"))
    
    # Matriz de confusão
    st.subheader("Matriz de Confusão")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Adimplente', 'Inadimplente'], yticklabels=['Adimplente', 'Inadimplente'])
    st.pyplot(fig)
    
    # Curva ROC
    st.subheader("Curva ROC")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'Área sob a curva (AUC) = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel("Falsos Positivos")
    ax.set_ylabel("Verdadeiros Positivos")
    ax.legend()
    st.pyplot(fig)
    
    # Importância das Features
    st.subheader("Importância das Features")
    importances = modelo.feature_importances_
    features = X.columns
    feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, ax=ax, palette='Blues_r')
    st.pyplot(fig)
