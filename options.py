import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import NearMiss

def preprocess_data(df):
    scaler = MinMaxScaler()
    df = df.copy()
    df['Nota da Clínica'] = df['Nota da Clínica'].apply(lambda x: 0 if x <= 3 else (1 if x <= 7 else 2))
    df['Total do Contrato (Bruto)/renda utilizada'] = df['Total do Contrato (Bruto)'] / df['renda utilizada']
    df.drop(columns=['Total do Contrato (Bruto)', 'renda utilizada'], inplace=True)
    
    X = df.drop(columns=['[SRM] Código da operação', 'variavel_target'])
    y = df['variavel_target']
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    nr = NearMiss()
    X, y = nr.fit_resample(X, y)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=0), scaler

def train_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(random_state=0)
    model.fit(X_train, y_train)
    return model

def train_neural_network(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_dim=X_train.shape[1]),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=0)
    return model

def train_catboost(X_train, y_train):
    model = CatBoostClassifier(verbose=0)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, scaler, model_type):
    if model_type == 'neural_network':
        y_pred = (model.predict(X_test) > 0.5).astype(int)
        y_proba = model.predict(X_test)
    else:
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    
    return report, conf_matrix, auc_score, fpr, tpr, y_proba

def plot_metrics(conf_matrix, auc_score, fpr, tpr):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Matriz de Confusão')
    axes[0].set_xlabel('Predito')
    axes[0].set_ylabel('Real')
    
    axes[1].plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
    axes[1].plot([0, 1], [0, 1], color='red', linestyle='--')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Curva ROC')
    axes[1].legend()
    
    st.pyplot(fig)

def main():
    st.title('Simulação de Modelos de Crédito')
    model_choice = st.sidebar.selectbox("Escolha o modelo", ['Tree Decision', 'Rede Neural', 'CatBoost'])
    
    inputs = {}
    input_names = ['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score',
                   'Carater Acoes', 'Carater Percentual Divida', 'Carater Restricoes', 'Serasa Protestos',
                   'VTM Valor Total', 'Taxa de Juros', 'Total do Contrato (Bruto)', 'renda utilizada']
    
    for name in input_names:
        inputs[name] = st.number_input(name, value=1.0 if 'Percentual' in name else 0, step=1.0)
    
    if st.button('Simular'):
        df = pd.read_csv('df_model.csv')  # Carregue o dataset original aqui
        (X_train, X_test, y_train, y_test), scaler = preprocess_data(df)
        
        model = None
        model_type = ''
        if model_choice == 'Tree Decision':
            model = train_decision_tree(X_train, y_train)
            model_type = 'tree'
        elif model_choice == 'Rede Neural':
            model = train_neural_network(X_train, y_train)
            model_type = 'neural_network'
        elif model_choice == 'CatBoost':
            model = train_catboost(X_train, y_train)
            model_type = 'catboost'
        
        report, conf_matrix, auc_score, fpr, tpr, y_proba = evaluate_model(model, X_test, y_test, scaler, model_type)
        
        st.write("### Relatório de Classificação")
        st.json(report)
        
        plot_metrics(conf_matrix, auc_score, fpr, tpr)
        
        user_input_df = pd.DataFrame([inputs])
        user_input_df['Nota da Clínica'] = user_input_df['Nota da Clínica'].apply(lambda x: 0 if x <= 3 else (1 if x <= 7 else 2))
        user_input_df['Total do Contrato (Bruto)/renda utilizada'] = user_input_df['Total do Contrato (Bruto)'] / user_input_df['renda utilizada']
        user_input_df.drop(columns=['Total do Contrato (Bruto)', 'renda utilizada'], inplace=True)
        user_input_df = pd.DataFrame(scaler.transform(user_input_df), columns=user_input_df.columns)
        
        prob_default = model.predict_proba(user_input_df)[:, 1] if model_type != 'neural_network' else model.predict(user_input_df)[0]
        
        st.write(f"### Probabilidade de Inadimplência: {prob_default[0]*100:.2f}%")
        color = 'green' if prob_default[0] < 0.5 else 'red'
        st.markdown(f'<p style="color:{color}; font-size:24px">{prob_default[0]*100:.2f}%</p>', unsafe_allow_html=True)
