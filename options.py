import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Função para converter Nota da Clínica
def transformar_nota(nota):
    if nota in [0, 1, 2, 3]:
        return 0
    elif nota in [4, 5, 6, 7]:
        return 1
    else:
        return 2

# Função para carregar dados (substitua pelo seu dataset)
def carregar_dados():
    df = pd.read_csv("seu_dataset.csv")
    return df

# Função para preprocessamento
def preprocessar_dados(df):
    X = df.drop("target", axis=1)
    y = df["target"]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

# Sidebar para navegação
st.sidebar.title("Simulação de Risco de Crédito")
pagina = st.sidebar.radio("Escolha o modelo:", ["Decision Tree", "Rede Neural"])

# Inputs do usuário
st.title(f"Simulação com {pagina}")

nota = st.number_input("Nota da Clínica (0 a 10)", min_value=0, max_value=10, step=1)
idade = st.number_input("Idade", min_value=18, max_value=100, step=1)
capital = st.number_input("Capital Endividamento")
serasa = st.number_input("Serasa Score")
acoes = st.number_input("Ações Judiciais / Cheques Sustados / PIE")
perc_divida = st.number_input("Percentual de Dívida Vencida Total")
restricoes = st.number_input("Quantidade de Restrições Comerciais")
protestos = st.number_input("Quantidade de Protestos")
vtm = st.number_input("Valor Total")
taxa_juros = st.number_input("Taxa de Juros")
total_contrato = st.number_input("Total do Contrato (Bruto)")
renda_utilizada = st.number_input("Renda Utilizada")

# Criando variável derivada
contrato_renda = total_contrato / renda_utilizada if renda_utilizada != 0 else 0

# Botão para simular
if st.button("Simular"):
    df = carregar_dados()
    X, y, scaler = preprocessar_dados(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    novo_dado = np.array([[transformar_nota(nota), idade, capital, serasa, acoes, perc_divida,
                            restricoes, protestos, vtm, taxa_juros, contrato_renda]])
    novo_dado = scaler.transform(novo_dado)
    
    if pagina == "Decision Tree":
        modelo = DecisionTreeClassifier()
    elif pagina == "Rede Neural":
        modelo = Sequential([
            Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        modelo.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)
    
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]
    
    # Exibir métricas
    st.text("Relatório de Classificação:")
    st.text(classification_report(y_test, y_pred))
    
    # Matriz de confusão
    st.text("Matriz de Confusão:")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)
    
    # Feature Importance
    if pagina in ["Decision Tree"]:
        importance = modelo.feature_importances_
        st.text("Importância das Features:")
        fig, ax = plt.subplots()
        sns.barplot(x=importance, y=df.drop("target", axis=1).columns)
        st.pyplot(fig)
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.text("Curva ROC:")
    fig, ax = plt.subplots()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("Falso Positivo")
    plt.ylabel("Verdadeiro Positivo")
    plt.legend()
    st.pyplot(fig)
    
    # Previsão para o novo cliente
    prob = modelo.predict_proba(novo_dado)[:, 1][0] * 100
    cor = "green" if prob < 50 else "red"
    st.markdown(f"<h3 style='color:{cor}'>Probabilidade de Inadimplência: {prob:.2f}%</h3>", unsafe_allow_html=True)
