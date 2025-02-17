import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from imblearn.under_sampling import NearMiss
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

@st.cache
def load_data():
    df_operacoes = pd.read_excel('base_operações.xlsx')
    variaveis_selecionadas = ['[SRM] Código da operação', "Total em Atraso", 'Total do Contrato (Bruto)', "Inad30", "Inad60", "Inad90", "Dias Vencidos", "Total Pago", "Total Cobrado", "Nota da Clínica", '[Capacidade] Idade', "[Capacidade] Situacao empregaticia", "[Capital] Endividamento", "[Serasa] Score", "[Carater] Acoes_Judiciais_Cheques_Sustados_e_PIE", "[Carater] Percentual_de_divida_vencida_total", "[Carater] Quantidade_de_restricoes_comerciais", "[Carater] Relevancia_das_restricoes_comerciais", "[Condicoes] Clinica", "[RELATORIO] Capacidade pontos", "[RELATORIO] Capital pontos", "[RELATORIO] Carater pontos", "[RELATORIO] Condicoes pontos", "[RELATORIO] PMT", "Risco", "[Serasa] chance de pagar", "[Serasa] Falencias", "[SERASA] Qte acoes judiciais", "[Serasa] QTE CCF", "[Serasa] qte Divida vencida", "[Serasa] Quantidade de protestos", "[VTM] Valor total", "renda utilizada", "Taxa de Juros"]
    df_operacoes_selecionadas = df_operacoes[variaveis_selecionadas]
    df_operacoes_selecionadas = df_operacoes_selecionadas.dropna(subset=['Total em Atraso'])

    def currency_to_float(currency_str):
        if isinstance(currency_str, str):
            return float(currency_str.replace('$', '').replace('R$', '').replace('R', '').replace(',', '').replace('.','').strip())
        return currency_str

    for col in ['Total em Atraso', 'Inad30', 'Inad60', 'Inad90', 'Total Pago', 'Total Cobrado', 'Total do Contrato (Bruto)']:
        df_operacoes_selecionadas[col] = df_operacoes_selecionadas[col].apply(currency_to_float)

    df_operacoes_selecionadas['Dias Vencidos'] = df_operacoes_selecionadas['Dias Vencidos'].fillna(0)

    def map_risk(risk_str):
        if isinstance(risk_str, str):
            if 'Baixo Risco' in risk_str:
                return 0
            elif 'Médio Risco' in risk_str:
                return 1
            elif 'Alto Risco' in risk_str:
                return 2
        return risk_str

    df_operacoes_selecionadas['Nota da Clínica'] = df_operacoes_selecionadas['Nota da Clínica'].apply(map_risk)
    df_operacoes_selecionadas = df_operacoes_selecionadas.dropna(subset=['[Capacidade] Idade'])

    def convert_to_int(age_str):
        if isinstance(age_str, str):
            return int(float(age_str.replace('.', '').replace(',', '.')))
        return age_str

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
    df_operacoes_selecionadas['[Capacidade] Situacao empregaticia'] = df_operacoes_selecionadas['[Capacidade] Situacao empregaticia'].apply(agrupar_situacao)
    df_dummies = pd.get_dummies(df_operacoes_selecionadas['[Capacidade] Situacao empregaticia']).astype(int)
    df_operacoes_selecionadas = pd.concat([df_operacoes_selecionadas, df_dummies], axis=1)
    df_operacoes_selecionadas.drop(columns=['[Capacidade] Situacao empregaticia'], inplace=True)
    df_operacoes_selecionadas = df_operacoes_selecionadas[df_operacoes_selecionadas['Total Cobrado'] > 0]

    df_operacoes_selecionadas['variavel_target'] = df_operacoes_selecionadas.apply(
        lambda row: 1 if row['Inad30'] > 0 or row['Inad60'] > 0 or row['Inad90'] > 0 or row['Dias Vencidos'] > 0 else 0,
        axis=1
    )

    df_model = df_operacoes_selecionadas[['[SRM] Código da operação', 'variavel_target', 'Total do Contrato (Bruto)', 'Nota da Clínica', '[Capacidade] Idade', '[Capital] Endividamento', '[Serasa] Score', '[Carater] Acoes_Judiciais_Cheques_Sustados_e_PIE', '[Carater] Percentual_de_divida_vencida_total', '[Carater] Quantidade_de_restricoes_comerciais', '[Carater] Relevancia_das_restricoes_comerciais', '[Condicoes] Clinica', '[RELATORIO] Capacidade pontos', '[RELATORIO] Capital pontos', '[RELATORIO] Carater pontos', '[RELATORIO] Condicoes pontos', '[RELATORIO] PMT', 'Risco', '[Serasa] chance de pagar', '[SERASA] Qte acoes judiciais', '[Serasa] Quantidade de protestos', '[VTM] Valor total', 'renda     utilizada', 'Taxa de Juros']]
    df_model = df_model.dropna(subset=['Nota da Clínica', 'Total do Contrato (Bruto)', '[Serasa] Score', '[Condicoes] Clinica', '[RELATORIO] Capacidade pontos', '[RELATORIO] Capital pontos', '[RELATORIO] Carater pontos', '[RELATORIO] Condicoes pontos', '[RELATORIO] PMT', '[Serasa] chance de pagar', '[VTM] Valor total', 'Taxa de Juros', 'renda utilizada'])
    df_model['[SERASA] Qte acoes judiciais'].fillna(0, inplace=True)
    df_model['[Serasa] Quantidade de protestos'].fillna(0, inplace=True)
    df_model['Total do Contrato (Bruto)/renda utilizada'] = df_model['Total do Contrato (Bruto)'] / df_model['renda utilizada']
    df_model = df_model[['[SRM] Código da operação', 'variavel_target', 'Total do Contrato (Bruto)', 'Nota da Clínica', '[Capacidade] Idade', '[Capital] Endividamento', '[Serasa] Score', '[Carater] Acoes_Judiciais_Cheques_Sustados_e_PIE', '[Carater] Percentual_de_divida_vencida_total', '[Carater] Quantidade_de_restricoes_comerciais', '[Serasa] Quantidade de protestos', '[VTM] Valor total', 'Taxa de Juros', 'Total do Contrato (Bruto)/renda utilizada']]
    df_model = df_model.dropna(subset=['Nota da Clínica', '[Serasa] Score', '[VTM] Valor total', 'Taxa de Juros'])
    df_model['[Serasa] Quantidade de protestos'].fillna(0, inplace=True)
    return df_model

df_model = load_data()

st.title('Simulação de Inadimplência')

# Sidebar para navegação
st.sidebar.title('Navegação')
page = st.sidebar.selectbox('Escolha a página', ['Decision Tree', 'Neural Network'])

# Função para converter a nota da clínica
def convert_clinic_score(score):
    if score in [0, 1, 2, 3]:
        return 0
    elif score in [4, 5, 6, 7]:
        return 1
    elif score in [8, 9, 10]:
        return 2

# Inputs do usuário
st.sidebar.header('Parâmetros de Entrada')
nota_clinica = st.sidebar.slider('Nota da Clínica (0 a 10)', 0, 10, 5)
capacidade_idade = st.sidebar.number_input('Idade', min_value=18, max_value=100, value=30)
capital_endividamento = st.sidebar.number_input('Endividamento', min_value=0, value=1000)
serasa_score = st.sidebar.number_input('Serasa Score', min_value=0, max_value=1000, value=500)
acoes_judiciais = st.sidebar.number_input('Ações Judiciais/Cheques Sustados/PIE', min_value=0, value=0)
percentual_divida_vencida = st.sidebar.number_input('Percentual de Dívida Vencida', min_value=0, max_value=100, value=0)
quantidade_restricoes = st.sidebar.number_input('Quantidade de Restrições Comerciais', min_value=0, value=0)
quantidade_protestos = st.sidebar.number_input('Quantidade de Protestos', min_value=0, value=0)
valor_total = st.sidebar.number_input('Valor Total (VTM)', min_value=0, value=1000)
taxa_juros = st.sidebar.number_input('Taxa de Juros (%)', min_value=0.0, max_value=100.0, value=5.0)
total_contrato = st.sidebar.number_input('Total do Contrato (Bruto)', min_value=0, value=1000)
renda_utilizada = st.sidebar.number_input('Renda Utilizada', min_value=0, value=1000)

# Calcular a variável Total do Contrato (Bruto)/renda utilizada
total_contrato_renda_utilizada = total_contrato / renda_utilizada if renda_utilizada != 0 else 0

# Converter a nota da clínica
nota_clinica_convertida = convert_clinic_score(nota_clinica)

# Função para treinar e avaliar o modelo
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    st.write("### Relatório de Classificação")
    st.text(classification_report(y_test, y_pred))
    st.write("### Matriz de Confusão")
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    st.pyplot()
    st.write("### Importância das Features")
    feature_importances = model.feature_importances_
    features = X_train.columns
    sns.barplot(x=feature_importances, y=features)
    st.pyplot()
    st.write("### Curva ROC")
    auc_score = roc_auc_score(y_test, y_proba)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    st.pyplot()

# Função para simular a probabilidade de inadimplência
def simulate_inadimplencia(model, input_data):
    proba = model.predict_proba(input_data)[0][1]
    st.write("### Probabilidade de Inadimplência")
    if proba < 0.5:
        st.markdown(f"<h1 style='color:green'>Probabilidade: {proba:.2f}</h1>", unsafe_allow_html=True)
    else:
        st.markdown(f"<h1 style='color:red'>Probabilidade: {proba:.2f}</h1>", unsafe_allow_html=True)

# Página Decision Treeif page == 'Decision Tree':
    st.header('Decision Tree Classifier')
    if st.sidebar.button('Simular'):
        # Pré-processamento dos dados
        scaler = MinMaxScaler()
        X = df_model.drop(columns=['[SRM] Código da operação', 'variavel_target', 'Total do Contrato (Bruto)'])
        y = df_model['variavel_target']
        X[['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score', 'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE', 'Carater Percentual_de_divida_vencida_total', 'Carater Quantidade_de_restricoes_comerciais', 'Serasa Quantidade de protestos', 'VTM Valor total', 'Taxa de Juros', 'Total do Contrato (Bruto)/renda utilizada']] = scaler.fit_transform(X[['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score', 'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE', 'Carater Percentual_de_divida_vencida_total', 'Carater Quantidade_de_restricoes_comerciais', 'Serasa Quantidade de protestos', 'VTM Valor total', 'Taxa de Juros', 'Total do Contrato (Bruto)/renda utilizada']])
        nr = NearMiss()
        X_resampled, y_resampled = nr.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, stratify=y_resampled, random_state=0)
        dt_model = DecisionTreeClassifier(random_state=0)
        param_grid = {'criterion': ['gini', 'entropy'], 'max_depth': [None, 10, 20, 30, 40, 50], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
        grid_search = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)
        best_dt_model = grid_search.best_estimator_
        train_and_evaluate_model(best_dt_model, X_train, y_train, X_test, y_test)
        input_data = pd.DataFrame([[nota_clinica_convertida, capacidade_idade, capital_endividamento, serasa_score, acoes_judiciais, percentual_divida_vencida, quantidade_restricoes, quantidade_protestos, valor_total, taxa_juros, total_contrato_renda_utilizada]], columns=X.columns)
        input_data = scaler.transform(input_data)
        simulate_inadimplencia(best_dt_model, input_data)
        # Página Neural Network
elif page == 'Neural Network':
    st.header('Neural Network Classifier')
    if st.sidebar.button('Simular'):
        # Pré-processamento dos dados
        scaler = MinMaxScaler()
        X = df_model.drop(columns=['[SRM] Código da operação', 'variavel_target', 'Total do Contrato (Bruto)'])
        y = df_model['variavel_target']
        X[['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score', 'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE', 'Carater Percentual_de_divida_vencida_total', 'Carater Quantidade_de_restricoes_comerciais', 'Serasa Quantidade de protestos', 'VTM Valor total', 'Taxa de Juros', 'Total do Contrato (Bruto)/renda utilizada']] = scaler.fit_transform(X[['Nota da Clínica', 'Capacidade Idade', 'Capital Endividamento', 'Serasa Score', 'Carater Acoes_Judiciais_Cheques_Sustados_e_PIE', 'Carater Percentual_de_divida_vencida_total', 'Carater Quantidade_de_restricoes_comerciais', 'Serasa Quantidade de protestos', 'VTM Valor total', 'Taxa de Juros', 'Total do Contrato (Bruto)/renda utilizada']])
        nr = NearMiss()
        X_resampled, y_resampled = nr.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.25, stratify=y_resampled, random_state=0)

        # Criar o modelo de rede neural
        nn_model = Sequential()
        nn_model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
        nn_model.add(Dense(32, activation='relu'))
        nn_model.add(Dense(1, activation='sigmoid'))

        # Compilar o modelo
        nn_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Treinar o modelo
        nn_model.fit(X_train, y_train, epochs=50, batch_size=10, verbose=1)

        # Fazer previsões no conjunto de teste
        y_pred = (nn_model.predict(X_test) > 0.5).astype("int32")
        y_proba = nn_model.predict(X_test)

        st.write("### Relatório de Classificação")
        st.text(classification_report(y_test, y_pred))
        st.write("### Matriz de Confusão")
        conf_matrix = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        st.pyplot()
        st.write("### Curva ROC")
        auc_score = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc_score:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc='lower right')
        st.pyplot()

        # Simular a probabilidade de inadimplência
        input_data = pd.DataFrame([[nota_clinica_convertida, capacidade_idade, capital_endividamento, serasa_score, acoes_judiciais, percentual_divida_vencida, quantidade_restricoes, quantidade_protestos, valor_total, taxa_juros, total_contrato_renda_utilizada]], columns=X.columns)
        input_data = scaler.transform(input_data)
        proba = nn_model.predict(input_data)[0][0]
        st.write("### Probabilidade de Inadimplência")
        if proba < 0.5:
            st.markdown(f"<h1 style='color:green'>Probabilidade: {proba:.2f}</h1>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h1 style='color:red'>Probabilidade: {proba:.2f}</h1>", unsafe_allow_html=True)
