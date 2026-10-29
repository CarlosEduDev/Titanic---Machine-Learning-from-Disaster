# 🚢 Titanic Survival Prediction - Random Forest Baseline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Kaggle Score](https://img.shields.io/badge/Kaggle%20Accuracy-0.77511-20BEFF.svg?logo=kaggle)](https://www.kaggle.com/c/titanic)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model%20Card-FFD21E.svg?logo=huggingface)](https://huggingface.co/EvaanCarloss/titanic-random-forest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Este repositório contém a implementação completa de uma solução de Machine Learning para a competição clássica do Kaggle **[Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)**. O objetivo é prever a taxa de sobrevivência dos passageiros com base em dados demográficos e socioeconômicos.

---

## 📌 Principais Resultados

* **Acurácia Oficial (Kaggle Public Score):** `0.77511` (77,51%)
* **Algoritmo:** Random Forest Classifier (`scikit-learn`)
* **Estratégia:** Baseline com pré-processamento de variáveis categóricas via *One-Hot Encoding*.

---

## 🛠️ Tecnologias e Bibliotecas

* **Linguagem:** Python 3.13.16
* **Manipulação de Dados:** `pandas`
* **Machine Learning:** `scikit-learn` (`RandomForestClassifier`)
* **Interface Gráfica & Deploy:** `gradio`

---

## 📂 Estrutura do Repositório

```bash
.
├── app.py                   # Aplicação interativa em Gradio para inferência local
├── requirements.txt         # Dependências do projeto para deploy e execução
├── titanic_competition.ipynb # Notebook com EDA, treino do modelo e geração do CSV
├── train.csv                # Conjunto de dados de treino fornecido pelo Kaggle
├── test.csv                 # Conjunto de dados de teste fornecido pelo Kaggle
└── submission.csv           # Arquivo final de submissão gerado para avaliação
