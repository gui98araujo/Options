import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
#from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from imblearn.under_sampling import NearMiss

# Configuração inicial do Streamlit
st.set_page_config(page_title="Análise de Crédito", layout="wide")

# Criar múltiplas páginas no Streamlit
pagina = st.sidebar.radio("Selecione o Modelo:", ["Tree Decision", "Rede Neural", #"CatBoost"])

# Inputs do usuário
st.header("Insira os dados do cliente")
nota_clinica = st.number_input("Nota da Clínica (0 a 10)", min_value=0, max_value=10, step=1)
capacidade_idade = st.number_input("Idade", min_value=18, max_value=100, step=1)
capital_endividamento = st.number_input("Endividamento", min_value=0.0, step=0.01)
serasa_score = st.number_input("Serasa Score", min_value=0, max_value=1000, step=1)
carater_acoes = st.number_input("Ações Judiciais, Cheques Sustados e PIE", min_value=0, step=1)
carater_divida = st.number_input("Percentual de Dívida Vencida", min_value=0.0, step=0.01)
carater_restricoes = st.number_input("Quantidade de Restrições Comerciais", min_value=0, step=1)
serasa_protestos = st.number_input("Quantidade de Protestos", min_value=0, step=1)
vtm_valor_total = st.number_input("Valor Total", min_value=0.0, step=0.01)
taxa_juros = st.number_input("Taxa de Juros", min_value=0.0, step=0.01)
renda_utilizada = st.number_input("Renda Utilizada", min_value=0.01, step=0.01)

# Transformação da Nota da Clínica
if nota_clinica in [0, 1, 2, 3]:
    nota_clinica = 0
elif nota_clinica in [4, 5, 6, 7]:
    nota_clinica = 1
else:
    nota_clinica = 2

total_contrato_renda = vtm_valor_total / renda_utilizada

# Criando DataFrame com os dados do usuário
dados_usuario = pd.DataFrame({
    "Nota da Clínica": [nota_clinica],
    "Capacidade Idade": [capacidade_idade],
    "Capital Endividamento": [capital_endividamento],
    "Serasa Score": [serasa_score],
    "Carater Acoes_Judiciais_Cheques_Sustados_e_PIE": [carater_acoes],
    "Carater Percentual_de_divida_vencida_total": [carater_divida],
    "Carater Quantidade_de_restricoes_comerciais": [carater_restricoes],
    "Serasa Quantidade de protestos": [serasa_protestos],
    "VTM Valor total": [vtm_valor_total],
    "Taxa de Juros": [taxa_juros],
    "Total do Contrato (Bruto)/renda utilizada": [total_contrato_renda]
})

# Botão para iniciar a simulação
if st.button("Simular"):
    # Carregar e processar os dados originais
    df = pd.read_csv("df_model.csv")  # Suponha que o dataset original esteja aqui
    X = df.drop(columns=["variavel_target"])
    y = df["variavel_target"]
    
    # Normalizar os dados
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    X = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Aplicar balanceamento com NearMiss
    nr = NearMiss()
    X_resampled, y_resampled = nr.fit_resample(X, y)
    
    # Divisão dos dados
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, random_state=0)
    
    if pagina == "Tree Decision":
        model = DecisionTreeClassifier()
    elif pagina == "Rede Neural":
        model = Sequential([
            Dense(64, activation='relu', input_dim=X_train.shape[1]),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
    #elif pagina == "CatBoost":
     #   model = CatBoostClassifier(verbose=0)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    st.subheader("1. Métricas do Modelo")
    st.text(classification_report(y_test, y_pred))
    
    st.subheader("2. Matriz de Confusão")
    conf_matrix = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig)
    
    st.subheader("3. Importância das Features")
    feature_importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else model.get_feature_importance()
    fig, ax = plt.subplots()
    sns.barplot(x=feature_importance, y=X.columns)
    st.pyplot(fig)
    
    st.subheader("4. Curva ROC")
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--')
    st.pyplot(fig)
    
    prob_inadimplencia = y_proba[0] * 100
    cor = "green" if prob_inadimplencia < 50 else "red"
    st.markdown(f"### 5. Probabilidade de Inadimplência: <span style='color:{cor};'>{prob_inadimplencia:.2f}%</span>", unsafe_allow_html=True)
