import os
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold

def preenche_tabela(dados):
    def preenche_na(rows):
        rows.loc[rows["ICU"] != 1] = rows.loc[rows["ICU"] != 1].fillna(method='bfill').fillna(method='ffill')
        return rows

    """
    Função para preenher os valores nulos do dataframe (dados)
    desconsiderando as informações das janelas que o paciente foi internado ICU != 1

    Retorna um dataFrame
    """

    #Identificando as colunas contínuas
    features_continuas_colunas = dados.iloc[:, 13:-2].columns
    #Fazendo o prenchimento dos valores nulos
    features_continuas = dados.groupby("PATIENT_VISIT_IDENTIFIER", as_index=False)[list(features_continuas_colunas) + ["ICU"]].apply(preenche_na)
    features_continuas.drop("ICU", axis=1, inplace=True)

    features_categoricas = dados.iloc[:, :13]
    saida = dados.iloc[:, -2:]

    dados_finais = pd.concat([features_categoricas, features_continuas, saida], ignore_index=True, axis=1)
    dados_finais.columns = dados.columns

    return dados_finais



def remove_corr_var(dados, valor_corte):
    """
    Função para remover as colunas do dataframe (dados) com valores de auto correlação
    superiores ao valor_corte informado

    Retorna um dataFrame 
    """

    matriz_corr = dados.iloc[:,3:-1].corr().abs()
    matriz_upper = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(np.bool))
    excluir = [coluna for coluna in matriz_upper.columns if any(matriz_upper[coluna] > valor_corte)]

    return dados.drop(excluir, axis=1)



def roda_n_modelos(modelo, dados, n):
    """
    Função para executar um modelo fazendo um split dos dados com train_test_split, n vezes 
    e gera uma média e o desvio padrão da roc_auc_score 

    Retorna dois valores: auc_medio, auc_std
    """

    x_columns = dados.columns
    y = dados['ICU']
    x = dados[x_columns].drop(["ICU"], axis=1)

    auc_lista = []

    for _ in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)

        modelo.fit(x_train, y_train)
        prob_predic = modelo.predict_proba(x_test)
        auc = roc_auc_score(y_test, prob_predic[:,1])
        auc_lista.append(auc)

    auc_medio = np.mean(auc_lista)
    auc_std = np.std(auc_lista)
    print(f"AUC médio {auc_medio}")
    print(f"Intervalo {auc_medio - 2* auc_std} - {auc_medio + 2* auc_std}")

    return auc_medio, auc_std  



def roda_modelo_cv(modelo, dados, n_splits, n_repeats):
    """
    Função para executar (modelo) fazendo um split dos (dados) com RepeatedStratifiedKFold
    com (n_splits) e executando (n_repeats)

    Returna um array com os resultados das seguintes métricas
       * accuracy
       * roc_auc
       * precision
       * recall
       * f1
       * average_precision'
       
       As métricas são aplicadas nos dados de teste e de treino
    """

    np.random.seed(689432)
    dados = dados.sample(frac=1).reset_index(drop=True)

    x_columns = dados.columns
    y = dados['ICU']
    x = dados[x_columns].drop(["ICU"], axis=1)

    cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)
    resultados = cross_validate(modelo, x, y, cv=cv, scoring=['accuracy','roc_auc', 'precision', 'recall', 'f1', 'average_precision'], return_train_score=True)

    #auc_medio = np.mean(resultados['test_roc_auc'])
    #auc_medio_treino = np.mean(resultados['train_roc_auc'])
    #print(f"AUC {auc_medio} - {auc_medio_treino}")
    # return auc_medio, auc_medio_treino, resultados
    return resultados

