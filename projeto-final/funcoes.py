import os
import sys

import pandas as pd
import numpy as np

### Função para preenher os valores nulos da tabela ###
def preenche_tabela(dados):
    def preenche_na(rows):
        rows.loc[rows["ICU"] != 1] = rows.loc[rows["ICU"] != 1].fillna(method='bfill').fillna(method='ffill')
        return rows

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


### Função para remover colunas da matriz de correlação com valores maiores que valor_corte ###
def remove_corr_var(dados, valor_corte):

    matriz_corr = dados.iloc[:,3:-1].corr().abs()
    matriz_upper = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(np.bool))
    excluir = [coluna for coluna in matriz_upper.columns if any(matriz_upper[coluna] > valor_corte)]

    return dados.drop(excluir, axis=1)