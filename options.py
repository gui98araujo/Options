import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.tree import DecisionTreeClassifier
#from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Função para transformação da nota
def transformar_nota(nota):
    if nota in [0, 1, 2, 3]:
        return 0
    elif nota in [4, 5, 6, 7]:
        return 1
    else:
        return 2

# Layout do Streamlit
st.set_page_config(page_title="Análise de Risco de Crédito", layout="wide")

# Sidebar para navegação
st.sidebar.title("Menu")
pagina = st.sidebar.radio("Selecione um Modelo", ["Rede Neural", "Decision Tree"])

# Inputs do usuário
st.title(f"Simulação com {pagina}")

nota = st.number_input("Nota da Clínica (1 a 10)", 1, 10, step=1)
idade = st.number_input("Idade", 18, 100, step=1)
endividamento = st.number_input("Endividamento (%)", 0.0, 100.0, step=0.1)
serasa_score = st.number_input("Serasa Score", 0, 1000, step=1)
acoes_judiciais = st.number_input("Ações Judiciais", 0, 50, step=1)
perc_divida = st.number_input("Percentual de Dívida Vencida (%)", 0.0, 100.0, step=0.1)
restricoes_comerciais = st.number_input("Restrições Comerciais", 0, 50, step=1)
protestos = st.number_input("Quantidade de Protestos", 0, 50, step=1)
vtm_valor_total = st.number_input("VTM Valor Total", 0.0, 1e6, step=1000.0)
taxa_juros = st.number_input("Taxa de Juros (%)", 0.0, 100.0, step=0.1)
valor_contrato = st.number_input("Valor do Contrato (Bruto)", 0.0, 1e6, step=1000.0)
renda_solicitante = st.number_input("Renda do Solicitante", 0.0, 1e6, step=1000.0)

# Criar dataframe com as features inseridas
dados_input = pd.DataFrame({
    'Nota Clínica': [transformar_nota(nota)],
    'Idade': [idade],
    'Endividamento': [endividamento],
    'Serasa Score': [serasa_score],
    'Ações Judiciais': [acoes_judiciais],
    'Percentual de Dívida Vencida': [perc_divida],
    'Restrições Comerciais': [restricoes_comerciais],
    'Quantidade de Protestos': [protestos],
    'VTM Valor Total': [vtm_valor_total],
    'Taxa de Juros': [taxa_juros],
    'Total Contrato/Renda': [valor_contrato / (renda_solicitante + 1e-6)]  # Evitar divisão por zero
})

# Simular banco de dados para treinamento
df_model = pd.DataFrame(np.random.rand(1000, len(dados_input.columns)), columns=dados_input.columns)
df_model['inadimplente'] = np.random.randint(0, 2, 1000)

# Separação de treino e teste
X = df_model.drop(columns=['inadimplente'])
y = df_model['inadimplente']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização dos dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
dados_input_scaled = scaler.transform(dados_input)

# Treinamento e Previsão
if st.button("Simular"):
    if pagina == "Rede Neural":
        modelo = Sequential([
            Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(8, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        modelo.fit(X_train, y_train, epochs=10, batch_size=10, verbose=0)
        y_pred = (modelo.predict(X_test) > 0.5).astype(int)
        prob_input = modelo.predict(dados_input_scaled)[0][0]
    
    #elif pagina == "CatBoost":
        #modelo = CatBoostClassifier(verbose=0)
        #modelo.fit(X_train, y_train)
        #y_pred = modelo.predict(X_test)
        #prob_input = modelo.predict_proba(dados_input_scaled)[0][1]
    
    elif pagina == "Decision Tree":
        modelo = DecisionTreeClassifier()
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)
        prob_input = modelo.predict_proba(dados_input_scaled)[0][1]
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    st.write(f"**Acurácia:** {acc:.2f}")
    st.write(f"**Precisão:** {prec:.2f}")
    st.write(f"**Recall:** {rec:.2f}")
    st.write(f"**F1-Score:** {f1:.2f}")

    # Matriz de Confusão
    cm = confusion_matrix(y_test, y_pred)
    fig = ff.create_annotated_heatmap(z=cm, x=['Não Inadimplente', 'Inadimplente'], 
                                      y=['Não Inadimplente', 'Inadimplente'], colorscale='Blues')
    st.plotly_chart(fig)
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    fig_roc = px.area(x=fpr, y=tpr, title=f"Curva ROC (AUC = {roc_auc:.2f})")
    st.plotly_chart(fig_roc)
    
    # Resultado final
    risco_cor = "red" if prob_input >= 0.5 else "green"
    st.markdown(f"### Probabilidade de inadimplência: <span style='color:{risco_cor}; font-size:20px;'>{prob_input:.2%}</span>", unsafe_allow_html=True)
