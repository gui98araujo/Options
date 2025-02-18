import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from imblearn.under_sampling import NearMiss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Função para transformar a nota da clínica
def transformar_nota(nota):
    if nota in [0, 1, 2, 3]:
        return 0
    elif nota in [4, 5, 6, 7]:
        return 1
    else:
        return 2

# Criando um dataset simulado
np.random.seed(42)
df = pd.DataFrame({
    'Nota_Clinica': np.random.randint(1, 11, 1000),
    'Idade': np.random.randint(18, 70, 1000),
    'Endividamento': np.random.uniform(0, 100, 1000),
    'Serasa_Score': np.random.randint(300, 1000, 1000),
    'Acoes_Judiciais': np.random.randint(0, 5, 1000),
    'Percentual_Divida_Vencida': np.random.uniform(0, 100, 1000),
    'Restricoes_Comerciais': np.random.randint(0, 5, 1000),
    'Quantidade_Protestos': np.random.randint(0, 5, 1000),
    'VTM_Valor_Total': np.random.uniform(1000, 50000, 1000),
    'Taxa_Juros': np.random.uniform(2, 15, 1000),
    'Valor_Contrato': np.random.uniform(5000, 100000, 1000),
    'Renda_Solicitante': np.random.uniform(1000, 20000, 1000),
    'Inadimplente': np.random.randint(0, 2, 1000)
})

# Aplicando transformações
df['Nota_Clinica'] = df['Nota_Clinica'].apply(transformar_nota)
df['Renda_Contrato'] = df['Valor_Contrato'] / df['Renda_Solicitante']
df.drop(columns=['Valor_Contrato', 'Renda_Solicitante'], inplace=True)

# Separando variáveis
X = df.drop(columns=['Inadimplente'])
y = df['Inadimplente']

# Normalizar os dados
scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Balanceamento
nr = NearMiss()
X_resampled, y_resampled = nr.fit_resample(X_scaled, y)

# Divisão treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, stratify=y_resampled, random_state=0)

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

# Botão de simulação
if st.sidebar.button('Simular'):
    tab1, tab2 = st.tabs(["Árvore de Decisão", "Rede Neural"])
    
    with tab1:
        st.header("Modelo: Árvore de Decisão")
        dt_model = DecisionTreeClassifier(random_state=0)
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)
        y_proba = dt_model.predict_proba(X_test)[:, 1]
        prob_inadimplencia = dt_model.predict_proba(dados_input_scaled)[:, 1][0] * 100
        
        st.write("### Métricas de Avaliação")
        st.text(classification_report(y_test, y_pred))
        
        st.write("### Matriz de Confusão")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        st.pyplot(fig)
        
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)
        st.write("### Curva ROC")
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
        ax.plot([0, 1], [0, 1], color='red', linestyle='--')
        st.pyplot(fig)
        
        st.write(f'**Probabilidade de Inadimplência:** {prob_inadimplencia:.2f}%')
    
    with tab2:
        st.header("Modelo: Rede Neural")
        model = Sequential()
        model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
        
        y_proba_nn = model.predict(X_test).flatten()
        y_pred_nn = (y_proba_nn > 0.5).astype(int)
        prob_inadimplencia_nn = model.predict(dados_input_scaled)[0][0] * 100
        
        st.write("### Métricas de Avaliação")
        st.text(classification_report(y_test, y_pred_nn))
        
        st.write(f'**Probabilidade de Inadimplência:** {prob_inadimplencia_nn:.2f}%')
