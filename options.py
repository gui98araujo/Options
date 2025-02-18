import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.neural_network import MLPClassifier
#from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# Função para transformar a nota da clínica
def transformar_nota(nota):
    if nota in [0, 1, 2, 3]:
        return 0
    elif nota in [4, 5, 6, 7]:
        return 1
    else:
        return 2

# Simulando um dataset
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

# Separando features e target
X = df.drop(columns=['Inadimplente'])
y = df['Inadimplente']

# Balanceamento de dados
X_0, y_0 = X[y == 0], y[y == 0]
X_1, y_1 = X[y == 1], y[y == 1]
X_1_resampled, y_1_resampled = resample(X_1, y_1, replace=True, n_samples=len(y_0), random_state=42)
X_balanced = pd.concat([X_0, X_1_resampled])
y_balanced = pd.concat([y_0, y_1_resampled])

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_balanced)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_balanced, test_size=0.2, random_state=42)

# Modelos
models = {
    "Rede Neural": MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=1000, random_state=42),
    #"CatBoost": CatBoostClassifier(verbose=0, random_state=42),
    "Tree Decision": DecisionTreeClassifier(random_state=42)
}

# Streamlit
st.title("Classificação de Inadimplência")
menu = st.sidebar.radio("Escolha um modelo", list(models.keys()))

# Inputs do usuário
st.header("Insira os dados para previsão")
user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(col, value=float(df[col].mean()))

if st.button("Simular"):
    model = models[menu]
    model.fit(X_train, y_train)
    
    # Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    y_input_proba = model.predict_proba(scaler.transform(pd.DataFrame([user_input])))[:, 1][0]
    
    # Métricas
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Exibindo métricas
    st.subheader("Métricas do Modelo")
    st.write(f"Acurácia: {acc:.2f}")
    st.write(f"Precisão: {prec:.2f}")
    st.write(f"Recall: {rec:.2f}")
    st.write(f"F1-Score: {f1:.2f}")
    
    # Matriz de Confusão
    st.subheader("Matriz de Confusão")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
    
    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader("Curva ROC")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'Área sob a curva (AUC) = {roc_auc:.2f}')
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('Falso Positivo')
    ax.set_ylabel('Verdadeiro Positivo')
    ax.legend()
    st.pyplot(fig)
    
    # DataFrame de resultados
    df_result = pd.DataFrame({
        'Probabilidade de Inadimplência': y_proba,
        'Classificação': ['Risco Alto' if p > 0.5 else 'Risco Baixo' for p in y_proba]
    })
    df_result = df_result.sort_values(by='Probabilidade de Inadimplência', ascending=False)
    
    st.subheader("Resultados da Classificação")
    def highlight_risk(val):
        color = 'red' if val > 0.5 else 'green'
        return f'background-color: {color}'
    st.dataframe(df_result.style.applymap(highlight_risk, subset=['Probabilidade de Inadimplência']))
    
    # Resultado do input
    st.subheader("Resultado do Caso Simulado")
    color = "🔴" if y_input_proba > 0.5 else "🟢"
    st.write(f"Probabilidade de inadimplência: {y_input_proba:.2f} {color}")
