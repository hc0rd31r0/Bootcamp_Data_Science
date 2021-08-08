# Funções auxiliares para o projeto final de data science
# by Helton Cordeiro e colegas de curso

import pandas as pd
import numpy as np
import time
import random

import matplotlib.pyplot as plt

from scipy.stats import randint

import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import classification_report
from sklearn.metrics import fbeta_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


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



def roda_modelo(model, dados):
  """
  Função para processar um model sobre os dados, utilizando train_test_split
  e mostrando o valor da AUC.

  A execução dessa função sofre interferência da aleatoriedade da função train_test_split
  para a separação dos dados.

  Retorna: auc
  """
  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU","WINDOW"], axis=1)

  x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)

  model.fit(x_train, y_train)
  predicao = model.predict(x_test)
  prob_predic = model.predict_proba(x_test)

  auc = roc_auc_score(y_test, prob_predic[:,1])
  print(f"AUC {auc}")
  print("\nClassification Report")
  print(classification_report(y_test, predicao))

  return auc



def roda_n_modelos(model, dados, n):
  """
  Essa função diminui os efeitos da aleatoriedade do train_test_split, veja roda_modelo(), 
  fazendo uma média da execução de n vezes do model sobre os dados.

  Retorna: auc_medio, auc_std
  """

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  auc_lista = []

  for _ in range(n):
      x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y)

      model.fit(x_train, y_train)
      prob_predic = model.predict_proba(x_test)
      auc = roc_auc_score(y_test, prob_predic[:,1])
      auc_lista.append(auc)

  auc_medio = np.mean(auc_lista)
  auc_std = np.std(auc_lista)
  print(f"AUC médio {auc_medio}")
  print(f"Intervalo {auc_medio - 2* auc_std} - {auc_medio + 2* auc_std}")

  return auc_medio, auc_std  



def roda_modelo_cv(model, dados, n_splits, n_repeats):
  """
  Função para executar (model) fazendo um split dos (dados) com RepeatedStratifiedKFold
  com (n_splits) e executando (n_repeats)

  Returna um array com os resultados das seguintes métricas
     * accuracy
     * roc_auc
     * average_precision
     
     Tannto dos dados teste e de treino
  """

  # Utilizamos Como o RepeatedStratifiedKFold não 
  np.random.seed(73246)
  dados = dados.sample(frac=1).reset_index(drop=True)

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)
#  resultados = cross_validate(model, x, y, cv=cv, scoring=['accuracy','roc_auc', 'precision', 'recall', 'f1', 'average_precision'], return_train_score=True)
  resultados = cross_validate(model, x, y, cv=cv, scoring=['accuracy','roc_auc', 'average_precision'], return_train_score=True)

  return resultados



def executa_modelos(names, models, dados, n_splits, n_repeats):
  """
  Executa uma lista de modelos (models), utilizando a função roda_modelo_cv(),
  passando n_splits e n_repeats para a roda_modelo_cv, e retorna um dataFrame
  com o resultado da média da Acurácia, Roc AUC, PR AUC, e ROC AUC do Treino

  Retorna: dataFrame ordenado por ROC AUC
  """

  dfretorno = pd.DataFrame()

  for name, model in zip(names, models):
    results = roda_modelo_cv(model, dados, n_splits, n_repeats)
    mean_results = {}
    for x in results:
      mean_results[x] = np.mean(results[x])

    df_result = pd.DataFrame([[name, model,
            mean_results['test_accuracy'],  mean_results['test_roc_auc'], mean_results['test_average_precision'],
            mean_results['train_accuracy'], mean_results['train_roc_auc'], mean_results['train_average_precision']]],
            columns=['Nome', 'Modelo', 'Accuracy', 'ROC AUC', 'PR AUC', 'Train Accuracy', 'Train ROC AUC', 'Train PR AUC'])

    dfretorno = dfretorno.append(df_result)

  dfretorno.reset_index(drop=True, inplace=True)
  dfretorno = dfretorno.set_index('Nome')
  dfretorno = dfretorno.sort_values(by='ROC AUC', ascending=False)

  return dfretorno



def roda_modelo_RandomizedSearchCV(model, dados, n_splits, n_repeats, param_distributions, n_iter):
  """
  Função para aplicar os hiperparametros (param_distributions) ao model, utilizando os dados,
  sendo separados em n_splits, sendo repetidos n_repeats, e o parâmetro n_iter é aplicado na iteração
  dos parâmetros.

  Returna: o resultado do RandomizedSearchCV, com as seguintes métricas:
     * accuracy
     * roc_auc
     * precision
     * recall
     * f1
     * average_precision
  """

  np.random.seed(73246)
  dados = dados.sample(frac=1).reset_index(drop=True)
  x_columns = dados.columns
  y = dados["ICU"]
  x = dados[x_columns].drop(["ICU"], axis=1)

  cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

  busca = RandomizedSearchCV(model, param_distributions=param_distributions,
                             n_iter=n_iter, cv=cv,
#                             scoring=['accuracy','roc_auc', 'precision', 'recall', 'f1', 'average_precision'], refit='roc_auc',
#                             return_train_score=True )
                             scoring='roc_auc', return_train_score=True )

  busca.fit(x, y)

  return busca



def executa_modelos_RandomizedSearchCV(names, models, dados, n_splits, n_repeats, param_distributions, n_iter, showMsg=True):
  """
  Executa teste dos hiperparametros dos modelos passados em param_distributions das informações nos 
  array names e models sobre os dados, utilizando a função roda_modelo_RandomizedSearchCV()
  """
  df_retorno_rand = pd.DataFrame()

  for name, model in zip(names, models):
    start_time = time.time()
    busca = roda_modelo_RandomizedSearchCV( model, dados, n_splits, n_repeats, param_distributions[name], n_iter)
    resultados = pd.DataFrame(busca.cv_results_)
    total_time = time.time() - start_time

    if showMsg:
      print(f'Modelo: {name} \t tempo: %s segundos' %int(total_time))

#    df_result2 = pd.DataFrame([[name, busca.best_estimator_, resultados.iloc[busca.best_index_]['mean_test_accuracy'], 
#                              resultados.iloc[busca.best_index_]['mean_test_roc_auc'], 
#                              resultados.iloc[busca.best_index_]['mean_test_average_precision'],
#                              resultados.iloc[busca.best_index_]['mean_train_roc_auc'], busca.best_params_, int(total_time)]],
#                            columns=['Nome', 'Modelo', 'Accuracy', 'ROC AUC', 'PR AUC', 'Train ROC AUC', 'Best Params', 'Tempo'])    

    df_result2 = pd.DataFrame([[name, busca.best_estimator_, resultados.iloc[busca.best_index_]['mean_test_score'], 
                              resultados.iloc[busca.best_index_]['mean_train_score'], 
                              resultados.iloc[busca.best_index_]['std_test_score'], busca.best_params_, int(total_time)]],
                            columns=['Nome', 'Modelo', 'AUC', 'Train AUC', 'Std AUC', 'Best Params', 'Tempo'])

    df_retorno_rand = df_retorno_rand.append(df_result2)

  df_retorno_rand.reset_index(drop=True, inplace=True)
  df_retorno_rand = df_retorno_rand.set_index('Nome')
  df_retorno_rand = df_retorno_rand.sort_values(by="AUC", ascending=False)

  return df_retorno_rand



def plotar_curva_roc_modelos(names, models, dados):
  """
   Função para plotar a curva ROC AUC da array de names e models, sobre os dados
  """
  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)

  plt.subplots(figsize=(8, 8))
  plt.title('Receiver Operating Characteristic')
  c=["blue","black","brown","red","yellow","green","orange","beige","turquoise","pink","cyan","magenta","SteelBlue"]
  i = 0
  for name, model in zip(names, models):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
    preds = probs[:,1]
    fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
    roc_auc = metrics.auc(fpr, tpr)

    label = (f'Model: {name}- AUC= %0.2f' %roc_auc)
    plt.plot(fpr, tpr, 'b', label = label, color=c[i])
    # plt.plot(fpr, tpr, 'b', label = label)
    i = i + 1


  plt.legend(bbox_to_anchor=(1.05,1), frameon=True,  fontsize='large', facecolor='Snow', shadow=True)
  plt.plot([0, 1], [0, 1],'r--')
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  plt.ylabel('True Positive Rate')
  plt.xlabel('False Positive Rate')
  plt.show()



def plotar_matrix_confusao(model, dados):
  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)
  model.fit(X_train, y_train)

  plot_confusion_matrix(model, X_test, y_test)



def plotar_matrix_confusao_modelos(names, models, dados, nrows, ncols):
  """
  Plotar a Matriz de Confusão do array de names e models, sobre os dados, em nrows e ncols

  Retorna um array dos modelos treinados
  """
  np.random.seed(73246)
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(13, 15))
  axis = []
  for ax in axes:
    axis.extend(ax)
  axes_ind = 0

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)

  trained_models = []
  for name, clf in zip(names, models):
    clf.fit(X_train, y_train)
    trained_models.append(clf)
    disp = plot_confusion_matrix(clf, X_test, y_test, ax=axis[axes_ind])
    axis[axes_ind].set_xlabel('Previsto', fontsize=12)
    axis[axes_ind].set_ylabel('Reais', fontsize=12)

    y_predict = clf.predict(X_test)
    y_predict_proba = clf.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_predict_proba[:,1])
    f1_score = fbeta_score(y_test, y_predict, average='macro', beta=1)
    axis[axes_ind].set_title(f'{name}, F1 = {f1_score:.2f}', fontsize=13)

    disp.im_.set_clim(0, 50)
    axes_ind += 1

  plt.tight_layout()
  plt.show()

  for ax in axis[len(names):]:
    ax.set_visible(False)
    fig.delaxes(ax)

  return trained_models




def montar_classificacao(names, models, dados):
  def format_classification_report(test, predict):
    df_cr = pd.DataFrame(classification_report(test, predict, output_dict=True)).T
    df_cr['precision']['accuracy'] = ''
    df_cr['recall']['accuracy'] = ''
    df_cr['support']['accuracy'] = df_cr['support']['macro avg']
    df_cr['support'] = df_cr['support'].astype('int32')
    return df_cr

    """
    Montar um dataFrame com o classification_report do array de names e models, sobre os dados
    """

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)

  dfs = []
  for name, model in zip(names, models):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    dfs.append(format_classification_report(y_test, y_predict))

  dffinal = pd.concat(dfs, axis=1, keys=names)
  return dffinal



def plotar_media_curva_roc(names, models, dados, nrows, ncols, n_splits=5, n_repeats=10):

  np.random.seed(73246)
  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)
  retorno = []

  for name, clf in zip(names, models):
    # plt.figure(figsize=(7,7))
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,100)
    i = 1
    for train,test in cv.split(x,y):
      prediction = clf.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
      fpr, tpr, t = metrics.roc_curve(y[test], prediction[:, 1])
      tprs.append(np.interp(mean_fpr, fpr, tpr))
      roc_auc = metrics.auc(fpr, tpr)
      aucs.append(roc_auc)
      # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
      plt.plot(fpr, tpr, lw=1, alpha=0.3, label='') 
      i= i+1

    plt.plot([0,1],[0,1],linestyle = '--', lw=2, color='red')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue',
      label=r'Mean ROC (AUC = %0.4f )' % (mean_auc), lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC {name}')
    plt.legend(bbox_to_anchor=(1.05,1), frameon=True,  fontsize='large', shadow=True)
    #plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    #plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    plt.show()
    valores = [name, mean_auc]
    retorno.append(valores)

  return retorno
