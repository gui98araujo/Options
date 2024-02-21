{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/gui98araujo/Options/blob/main/options.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "MgiVNgOcuS8w"
      },
      "outputs": [],
      "source": [
        "import streamlit as st\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "\n",
        "# Configurações de estilo do seaborn\n",
        "sns.set_style(\"whitegrid\")\n",
        "\n",
        "# Função para carregar e processar os dados do CSV\n",
        "def carregar_dados(tipo_ativo):\n",
        "    if tipo_ativo == \"Açúcar\":\n",
        "        data = pd.read_csv('Dados Históricos - Açúcar NY nº11 Futuros (6).csv')\n",
        "        valor_minimo_padrao = 20.0\n",
        "        limite_inferior = 15\n",
        "        limite_superior = 35\n",
        "    elif tipo_ativo == \"Dólar\":\n",
        "        data = pd.read_csv('USD_BRL Dados Históricos (2).csv')\n",
        "        valor_minimo_padrao = 5.0\n",
        "        limite_inferior = 4\n",
        "        limite_superior = 6\n",
        "\n",
        "    data = data.rename(columns={'Último': 'Close', 'Data': 'Date'})\n",
        "    data['Date'] = pd.to_datetime(data['Date'], format='%d.%m.%Y')\n",
        "    data = data.sort_values(by='Date', ascending=True)\n",
        "    data['Close'] = data['Close'].str.replace(',', '.').astype(float)\n",
        "    data['Daily Return'] = data['Close'].pct_change()\n",
        "\n",
        "    return data, valor_minimo_padrao, limite_inferior, limite_superior\n",
        "\n",
        "# Função para simulação Monte Carlo\n",
        "def simulacao_monte_carlo(media_retornos_diarios, desvio_padrao_retornos_diarios, dias_simulados, num_simulacoes, limite_inferior, limite_superior, valor_strike):\n",
        "    retornos_diarios_simulados = np.random.normal(media_retornos_diarios, desvio_padrao_retornos_diarios, (dias_simulados, num_simulacoes))\n",
        "\n",
        "    preco_inicial = data['Close'].iloc[-1]\n",
        "    precos_simulados = np.ones((dias_simulados + 1, num_simulacoes)) * preco_inicial\n",
        "\n",
        "    valor_opcao = 0\n",
        "\n",
        "    for dia in range(1, dias_simulados + 1):\n",
        "        precos_simulados[dia, :] = precos_simulados[dia - 1, :] * (1 + retornos_diarios_simulados[dia - 1, :])\n",
        "        precos_simulados[dia, :] = np.maximum(np.minimum(precos_simulados[dia, :], limite_superior), limite_inferior)\n",
        "\n",
        "        valor_opcao += np.sum(np.maximum(precos_simulados[dia, :] - valor_strike, 0))\n",
        "\n",
        "    valor_justo = valor_opcao / (num_simulacoes * dias_simulados)\n",
        "    return valor_justo\n",
        "\n",
        "# Função para simulação das calls\n",
        "def simular_calls(dias_simulados, data, valor_minimo_padrao, limite_inferior, limite_superior):\n",
        "    media_retornos_diarios = data['Daily Return'].mean()\n",
        "    desvio_padrao_retornos_diarios = data['Daily Return'].std()\n",
        "    num_simulacoes = 100000  # Alteração para 100000 simulações\n",
        "\n",
        "    # Realizar a simulação para diferentes valores de preço da call\n",
        "    precos_calls = []\n",
        "    for preco_call in np.arange(limite_inferior, limite_superior + 0.25, 0.25):\n",
        "        valor_justo = simulacao_monte_carlo(media_retornos_diarios, desvio_padrao_retornos_diarios, dias_simulados, num_simulacoes, limite_inferior, limite_superior, preco_call)\n",
        "        precos_calls.append([preco_call, valor_justo])\n",
        "    return precos_calls\n",
        "\n",
        "# Configuração do título do aplicativo Streamlit e remoção da barra lateral\n",
        "st.set_page_config(page_title=\"Simulação de Preços de Calls\", page_icon=\"📈\", layout=\"wide\")\n",
        "\n",
        "# Título do sidebar\n",
        "st.sidebar.title('Simulação de Preços de Calls')\n",
        "\n",
        "# Input dos valores desejados\n",
        "tempo_desejado = st.sidebar.slider(\"Para quantos dias você quer avaliar o preço?\", min_value=1, max_value=360, value=30)\n",
        "\n",
        "# Botão para simular\n",
        "if st.sidebar.button(\"Simular\"):\n",
        "    # Carregar os dados\n",
        "    data, valor_minimo_padrao, limite_inferior, limite_superior = carregar_dados(\"Açúcar\")\n",
        "\n",
        "    # Simulação das calls\n",
        "    resultados = simular_calls(tempo_desejado, data, valor_minimo_padrao, limite_inferior, limite_superior)\n",
        "\n",
        "    # Exibição dos resultados em forma de tabela\n",
        "    st.write(\"Preços das Calls para\", tempo_desejado, \"dias:\")\n",
        "    df_resultados = pd.DataFrame(resultados, columns=[\"Preço da Call\", \"Valor Justo\"])\n",
        "    st.write(df_resultados)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "Qtek1sV82PwV",
        "outputId": "ba1124de-9915-4b40-870d-a45046426dd5"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.4/8.4 MB\u001b[0m \u001b[31m17.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m195.4/195.4 kB\u001b[0m \u001b[31m20.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.8/4.8 MB\u001b[0m \u001b[31m38.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m83.0/83.0 kB\u001b[0m \u001b[31m10.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m7.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h35.185.156.97\n",
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to False.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://35.185.156.97:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[K\u001b[?25hnpx: installed 22 in 5.769s\n",
            "your url is: https://nine-corners-deny.loca.lt\n"
          ]
        }
      ],
      "source": [
        "\n",
        "#! pip install streamlit -q\n",
        "#!wget -q -O - ipv4.icanhazip.com\n",
        "#! streamlit run options.py & npx localtunnel --port 8501"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZMZbZMsn2Zcd"
      },
      "outputs": [],
      "source": []
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPOrMhreA1xMROIIZ1SAkFd",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}