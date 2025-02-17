import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.under_sampling import NearMiss
#from catboost import CatBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Função para converter string de moeda para float
def currency_to_float(currency_str):
    if isinstance(currency_str, str):
        return float(currency_str.replace('$', '').replace('R$', '').replace('R', '').replace(',', '').replace('.','').strip())
    return currency_str

# Função para mapear os valores de risco para numérico
def map_risk(risk_str):
    if isinstance(risk_str, str):
        if 'Baixo Risco' in risk_str:
            return 0
        elif 'Médio Risco' in risk_str:
            return 1
        elif 'Alto Risco' in risk_str:
            return 2
    return risk_str

# Função para converter string de idade para inteiro
def convert_to_int(age_str):
    if isinstance(age_str, str):
        return int(float(age_str.replace('.', '').replace(',', '.')))
    return age_str

# Função para agrupar categorias redundantes
def agrupar_situacao(situacao):
    if situacao in ['AUTÔNOMO', 'SÓCIO', 'PROFISSIONAL LIBERAL']:
        return 'TRABALHADOR INDEPENDENTE'
    elif situacao in ['EMPREGADO CLT', 'FUNC PUB - CONTRATADO']:
        return 'EMPREGADO'
    elif situacao == 'APOSENTADO':
        return 'APOSENTADO'
    elif situacao == 'DESEMPREGADO / FUNC. PÚBLICO COMISSIONADO':
        return 'DESEMPREGADO'
    else:
        return situacao

# Função para normalizar os dados e aplicar NearMiss para balanceamento dos dados
def preprocess_data(df_model):
    scaler = MinMaxScaler()
    X = df_model.drop(columns=['[SRM] Código da operação','variavel_target','Total do Contrato (Bruto)'])
    y = df_model['variavel_target']

    X[['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score',
       'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE', 'Carater Percentual_de_divida_vencida_total',
       'Carater Quantidade_de_restricoes_comerciais', 'Serasa Quantidade de protestos', 'VTM Valor total',
       'Taxa de Juros', 'Total do Contrato (Bruto)/renda utilizada']] = scaler.fit_transform(
        X[['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score',
           'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE', 'Carater Percentual_de_divida_vencida_total',
           'Carater Quantidade_de_restricoes_comerciais', 'Serasa Quantidade de protestos', 'VTM Valor total',
           'Taxa de Juros', 'Total do Contrato (Bruto)/renda utilizada']]
    )

    nr = NearMiss()
    X_resampled, y_resampled = nr.fit_resample(X, y)
    
    return X_resampled, y_resampled, scaler

# Função para treinar e avaliar o modelo Decision Tree Classifier
def train_decision_tree(X_train, y_train, X_test, y_test):
    dt_model = DecisionTreeClassifier(random_state=0)
    param_grid = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30, 40, 50],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    best_dt_model = grid_search.best_estimator_
    
    dt_y_pred = best_dt_model.predict(X_test)
    dt_y_proba = best_dt_model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, dt_y_pred)
    conf_matrix = confusion_matrix(y_test, dt_y_pred)
    
    auc_score = roc_auc_score(y_test, dt_y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, dt_y_proba)
    
    feature_importances = best_dt_model.feature_importances_
    
    return report, conf_matrix, auc_score, fpr, tpr, feature_importances

# Função para treinar e avaliar o modelo CatBoostClassifier
def train_catboost(X_train, y_train, X_test, y_test):
    model = CatBoostClassifier()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    
    feature_importances = model.get_feature_importance()
    
    return report, conf_matrix, auc_score, fpr, tpr, feature_importances

# Função para treinar e avaliar o modelo Neural Network
def train_neural_network(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
    
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    _score = roc_auc_score(y_test, model.predict(X_test))
    fpr, tpr, thresholds = roc_curve(y_test, model.predict(X_test))
    
    feature_importances = None
    
    return report, conf_matrix, auc_score, fpr, tpr

# Função para exibir os resultados no Streamlit
def display_results(report, conf_matrix, auc_score, fpr, tpr):
    st.text(report)
    
    st.write("Matriz de Confusão:")
    fig_conf_matrix = plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    st.pyplot(fig_conf_matrix)
    
    st.write("Curva ROC:")
    fig_roc_curve = plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    st.pyplot(fig_roc_curve)

# Função principal para o Streamlit
def main():
    st.title("Simulação de Modelos de Machine Learning")
    
    menu = ["Decision Tree", "Rede Neural", "CatBoost"]
    choice = st.sidebar.selectbox("Selecione o Modelo", menu)
    
    st.sidebar.write("Insira os dados do cliente:")
    nota_clinica = st.sidebar.slider("Nota da Clínica (0 a 10)", 0, 10, 5)
    capacidade_idade = st.sidebar.number_input("Capacidade Idade", min_value=18, max_value=100, value=30)
    capital_endividamento = st.sidebar.number_input("Capital Endividamento", value=0.0)
    serasa_score = st.sidebar.number_input("Serasa Score", value=0.0)
    carater_acoes = st.sidebar.number_input("Carater Acoes Judiciais Cheques Sustados e PIE", value=0.0)
    carater_percentual = st.sidebar.number_input("Carater Percentual de Divida Vencida Total", value=0.0)
    carater_quantidade = st.sidebar.number_input("Carater Quantidade de Restricoes Comerciais", value=0.0)
    serasa_quantidade = st.sidebar.number_input("Serasa Quantidade de Protestos", value=0.0)
    vtm_valor_total = st.sidebar.number_input("VTM Valor Total", value=0.0)
    taxa_juros = st.sidebar.number_input("Taxa de Juros", value=0.0)
    total_contrato = st.sidebar.number_input("Total do Contrato (Bruto)", value=0.0)
    renda_utilizada = st.sidebar.number_input("Renda Utilizada", value=0.0)
    
    # Transformar Nota da Clínica
    if nota_clinica in [0, 1, 2, 3]:
        nota_clinica = 0
    elif nota_clinica in [4, 5, 6, 7]:
        nota_clinica = 1
    else:
        nota_clinica = 2
    
    # Calcular Total do Contrato (Bruto)/renda utilizada
    total_contrato_renda = total_contrato / renda_utilizada if renda_utilizada != 0 else 0
    
    # Dados de entrada
    input_data = {
        'Nota da Clínica': [nota_clinica],
        'Capacidade Idade': [capacidade_idade],
        'Capital Endividamento': [capital_endividamento],
        'Serasa Score': [serasa_score],
        'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE': [carater_acoes],
        'Carater Percentual_de_divida_vencida_total': [carater_percentual],
        'Carater Quantidade_de_restricoes_comerciais': [carater_quantidade],
        'Serasa Quantidade de protestos': [serasa_quantidade],
        'VTM Valor total': [vtm_valor_total],
        'Taxa de Juros': [taxa_juros],
        'Total do Contrato (Bruto)/renda utilizada': [total_contrato_renda]
    }
    
    df_input = pd.DataFrame(input_data)
    
    if st.sidebar.button("Simular"):
        # Carregar e preprocessar os dados
        df_operacoes = pd.read_excel('base_operações.xlsx')
        df_operacoes_selecionadas = df_operacoes[variaveis_selecionadas]
        df_operacoes_selecionadas = df_operacoes_selecionadas.dropna(subset=['Total em Atraso'])
        
        for col in ['Total em Atraso', 'Inad30', 'Inad60', 'Inad90', 'Total Pago', 'Total Cobrado', 'Total do Contrato (Bruto)']:
            df_operacoes_selecionadas[col] = df_operacoes_selecionadas[col].apply(currency_to_float)
        
        df_operacoes_selecionadas['Dias Vencidos'] = df_operacoes_selecionadas['Dias Vencidos'].fillna(0)
        df_operacoes_selecionadas['Nota da Clínica'] = df_operacoes_selecionadas['Nota da Clínica'].apply(map_risk)
        df_operacoes_selecionadas = df_operacoes_selecionadas.dropna(subset=['[Capacidade] Idade'])
        
        df_operacoes_selecionadas['[Capacidade] Idade'] = df_operacoes_selecionadas['[Capacidade] Idade'].apply(convert_to_int)
        df_operacoes_selecionadas['[Capital] Endividamento'] = df_operacoes_selecionadas['[Capital] Endividamento'].apply(convert_to_int)
        df_operacoes_selecionadas['[Carater] Percentual_de_divida_vencida_total'] = df_operacoes_selecionadas['[Carater] Percentual_de_divida_vencida_total'].apply(convert_to_int)
        df_operacoes_selecionadas['[Carater] Relevancia_das_restricoes_comerciais'] = df_operacoes_selecionadas['[Carater] Relevancia_das_restricoes_comerciais'].apply(convert_to_int)
        df_operacoes_selecionadas['[RELATORIO] Capacidade pontos'] = df_operacoes_selecionadas['[RELATORIO] Capacidade pontos'].apply(convert_to_int)
        df_operacoes_selecionadas['[RELATORIO] Capital pontos'] = df_operacoes_selecionadas['[RELATORIO] Capacidade pontos'].apply(convert_to_int)
        df_operacoes_selecionadas['[RELATORIO] Carater pontos'] = df_operacoes_selecionadas['[RELATORIO] Carater pontos'].apply(convert_to_int)
        df_operacoes_selecionadas['[RELATORIO] Condicoes pontos'] = df_operacoes_selecionadas['[RELATORIO] Condicoes pontos'].apply(convert_to_int)
        df_operacoes_selecionadas['[RELATORIO] PMT'] = df_operacoes_selecionadas['[RELATORIO] PMT'].apply(convert_to_int)
        df_operacoes_selecionadas['Risco'] = df_operacoes_selecionadas['Risco'].apply(convert_to_int)
        df_operacoes_selecionadas['[Serasa] chance de pagar'] = df_operacoes_selecionadas['[Serasa] chance de pagar'].apply(convert_to_int)
        df_operacoes_selecionadas['[VTM] Valor total'] = df_operacoes_selecionadas['[VTM] Valor total'].apply(convert_to_int)
        df_operacoes_selecionadas['renda utilizada'] = df_operacoes_selecionadas['renda utilizada'].apply(convert_to_int)
        df_operacoes_selecionadas['Taxa de Juros'] = df_operacoes_selecionadas['Taxa de Juros'].apply(convert_to_int)
        
        df_operacoes_selecionadas['[Capacidade] Situacao empregaticia'] = df_operacoes_selecionadas['[Capacidade] Situacao empregaticia'].apply(agrupar_situacao)
        df_dummies = pd.get_dummies(df_operacoes_selecionadas['[Capacidade] Situacao empregaticia']).astype(int)
        df_operacoes_selecionadas = pd.concat([df_operacoes_selecionadas, df_dummies], axis=1)
        df_operacoes_selecionadas.drop(columns=['[Capacidade] Situacao empregaticia'], inplace=True)
        
        df_operacoes_selecionadas = df_operacoes_selecionadas[df_operacoes_selecionadas['Total Cobrado'] > 0]
        
        df_operacoes_selecionadas['variavel_target'] = df_operacoes_selecionadas.apply(
            lambda row: 1 if row['Inad30'] > 0 or row['Inad60'] > 0 or row['Inad90'] > 0 or row['Dias Vencidos'] > 0 else 0,
            axis=1
        )
        
        df_model = df_operacoes_selecionadas[['[SRM] Código da operação','variavel_target','Total do Contrato (Bruto)','Nota da Clínica', '[Capacidade] Idade', '[Capital] Endividamento', 
                                              '[Serasa] Score','[Carater] Acoes_Judiciais_Cheques_Sustados_e_PIE',
                                              '[Carater] Percentual_de_divida_vencida_total', '[Carater] Quantidade_de_restricoes_comerciais',
                                              '[Serasa] Quantidade de protestos', '[VTM] Valor total', 'Taxa de Juros',
                                              'Total do Contrato (Bruto)/renda utilizada']]
        
        df_model = df_model.dropna(subset=['Nota da Clínica', 'Total do Contrato (Bruto)', '[Serasa] Score', '[VTM] Valor total', 'Taxa de Juros', 'renda utilizada'])
        df_model['[Serasa] Quantidade de protestos'].fillna(0, inplace=True)
        
        X_resampled, y_resampled, scaler = preprocess_data(df_model)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, stratify=y_resampled, random_state=0)
        
        if choice == "Decision Tree":
            report, conf_matrix, auc_score, fpr, tpr, feature_importances = train_decision_tree(X_train, y_train, X_test, y_test)
            display_results(report, conf_matrix, auc_score, fpr, tpr)
            st.write("Importância das Features:")
            fig_feature_importance = plt.figure(figsize=(10, 7))
            sns.barplot(x=feature_importances, y=X_train.columns)
            st.pyplot(fig_feature_importance)
        
        elif choice == "CatBoost":
            report, conf_matrix, auc_score, fpr, tpr, feature_importances = train_catboost(X_train, y_train, X_test, y_test)
            display_results(report, conf_matrix, auc_score, fpr, tpr)
            st.write("Importância das Features:")
            fig_feature_importance = plt.figure(figsize=(10, 7))
            sns.barplot(x=feature_importances, y=X_train.columns)
            st.pyplot(fig_feature_importance)
        
        elif choice == "Rede Neural":
            report, conf_matrix, auc_score, fpr, tpr = train_neural_network(X_train, y_train, X_test, y_test)
            display_results(report, conf_matrix, auc_score, fpr, tpr)
        
        # Fazer previsões com os dados de entrada
        df_input_normalized = scaler.transform(df_input)
        if choice == "Decision Tree":
            model = DecisionTreeClassifier(random_state=0)
            model.fit(X_train, y_train)
            proba = model.predict_proba(df_input_normalized)[:, 1]
        elif choice == "CatBoost":
            model = CatBoostClassifier()
            model.fit(X_train, y_train)
            proba = model.predict_proba(df_input_normalized)[:, 1]
        elif choice == "CatBoost":
            model = CatBoostClassifier()
            model.fit(X_train, y_train)
            proba = model.predict_proba(df_input_normalized)[:, 1]
        elif choice == "Rede Neural":
            model = Sequential()
            model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)
            proba = model.predict(df_input_normalized)[0]

        # Exibir a probabilidade de inadimplência
        st.write("Probabilidade de Inadimplência:")
        if proba >= 0.5:
            st.markdown(f"<h1 style='color: red;'>{proba:.2f}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color: green;'>{proba:.2f}</h1>", unsafe_allow_html=True)

if __name__ == '__main__':
    main()
