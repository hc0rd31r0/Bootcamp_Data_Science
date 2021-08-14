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
from sklearn.metrics import f1_score
from sklearn.metrics import plot_confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import RandomizedSearchCV


def preenche_tabela(dados):
  """
  
  Função para preenher os valores nulos do dataframe desconsiderando as informações 
  das janelas de atendimento que o paciente foi internado (ICU = 1)

  Parâmetros
  ----------
    dados: dataFrame.

  Retorno
  -------
    dfretorno: dataFrame com o resultado do processamento.

  """

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

  dfretorno = pd.concat([features_categoricas, features_continuas, saida], ignore_index=True, axis=1)
  dfretorno.columns = dados.columns

  return dfretorno



def remove_corr_var(dados, valor_corte= 0.95):
  """
  
  Função para remover as colunas do dataframe com valor de correlação.
  
  Parâmetros
  ----------
    dados: dataFrame.
    valor_corte: Valor de corte da correlação, default=0.95

  Retorno
  -------
    dfretorno: dataFrame com o resultado do processamento.

  """

  matriz_corr = dados.iloc[:,3:-1].corr().abs()
  matriz_upper = matriz_corr.where(np.triu(np.ones(matriz_corr.shape), k=1).astype(np.bool))
  excluir = [coluna for coluna in matriz_upper.columns if any(matriz_upper[coluna] > valor_corte)]

  return dados.drop(excluir, axis=1)



def roda_modelo(model, dados):
  """

  Função para processar um model sobre os dados, utilizando train_test_split e retornardo o valor da AUC.

  A execução dessa função sofre interferência da aleatoriedade da função train_test_split para a separação dos dados.

  Parâmetros
  ----------
    modelo: modelo a ser executado.
    dados: fonte de dados

  Retorno
  -------
    auc: resultado da função roc_auc_score()

  """
  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU","WINDOW"], axis=1)

  x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)

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

  Parâmetros
  ----------
    modelo: modelo a ser executado.
    dados: fonte de dados
    n: número de execuções para obter a média.

  Retorno
  -------
    auc: resultado da função roc_auc_score()
    
  Essa função diminui os efeitos da aleatoriedade do train_test_split, veja roda_modelo(), 
  fazendo uma média da execução de n vezes do model sobre os dados.

  Retorna: auc_medio, auc_std
  """
  
  np.random.seed(73246)
  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  auc_lista = []

  for _ in range(n):
      x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)

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
  
  Função que processa um modelo com os dados utilizando RepeatedStratifiedKFold()

  Parâmetros
  ----------
    model: modelo a ser executado.
    dados: fonte de dados
    n_splits: quantidade de splits de dados usado pela função RepeatedStratifiedKFold()
    n_repeats: quantidade de repeats usado pela função RepeatedStratifiedKFold()

  Retorno
  -------
    resultados: resultado do cross_validate()

  """

  np.random.seed(73246)
  dados = dados.sample(frac=1).reset_index(drop=True)

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  cv = RepeatedStratifiedKFold(n_splits = n_splits, n_repeats = n_repeats)
  resultados = cross_validate(model, x, y, cv=cv, scoring=['accuracy','roc_auc', 'average_precision'], return_train_score=True)

  return resultados



def executa_modelos(names, models, dados, n_splits, n_repeats):
  """

  Processa uma lista de modelos utilizando a função roda_modelo_cv(),
  e retorna um dataFrame como resultado.

  Parâmetros
  ----------
    names: array com os nomes dos modelos, e também é usado como o index do dataFrame de retorno.
             ex.: names = [ "KNeighbors", "Gaussian" ]
    models: array com as instâncias dos modelos a serem processados.
             ex.: classes = [ KNeighborsClassifier(), 
                              GaussianProcessClassifier() ]
    dados: dataFrame com as variáveis a serem analisadas
    n_splits: quantidade de splits de dados usado pela função RepeatedStratifiedKFold()
    n_repeats: quantidade de repeats usado pela função RepeatedStratifiedKFold()

  Retorno
  -------
    dfretorno: dataFrame com os resultados

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
  # dfretorno = dfretorno.sort_values(by='ROC AUC', ascending=False)

  return dfretorno



def roda_modelo_RandomizedSearchCV(model, dados, n_splits, n_repeats, param_distributions, n_iter):
  """
  
  Função para aplicar os hiperparametros (param_distributions) ao modelo.
  
  Parâmetros
  ----------
    model: Modelo de Machine Learning.
    dados: dataFrame com os dados
    n_splits: quantidade de splits de dados usado pela função RepeatedStratifiedKFold
    n_repeats: quantidade de repeats usado pela função RepeatedStratifiedKFold
    param_distributions: parâmetros a serem testados
    n_iter: Número de configurações de parâmetro que serão testados.

  Retorno
  -------
    busca: objeto RandomizedSearchCV
  
  """

  np.random.seed(73246)
  dados = dados.sample(frac=1).reset_index(drop=True)

  x_columns = dados.columns
  y = dados["ICU"]
  x = dados[x_columns].drop(["ICU"], axis=1)

  cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

  busca = RandomizedSearchCV(estimator=model, param_distributions=param_distributions,
                             n_iter=n_iter, cv=cv,
#                             scoring=['accuracy','roc_auc', 'precision', 'recall', 'f1', 'average_precision'], refit='roc_auc',
                             scoring='roc_auc', 
                             return_train_score=True, n_jobs=-1, random_state=73246)
  busca.fit(x, y)

  return busca



def executa_modelos_RandomizedSearchCV(names, models, dados, n_splits, n_repeats, param_distributions, n_iter, showMsg=True):
  """
  
  Função que recebe parâmetros para a execução de teste de hiperparametros dos modelos.
  Utiliza a função roda_modelo_RandomizedSearchCV()
  
  Parâmetros
  ----------
    names: array com os nomes dos modelos. É usado como o index do dataFrame de retorno.
             ex.: names = [ "KNeighbors", "Gaussian" ]
    models: array com a instância do modelo a ser testado.
             ex.: classes = [ KNeighborsClassifier(), 
                              GaussianProcessClassifier() ]
    dados: dataFrame com os dados
    n_splits: parâmetro utilizado pelo roda_modelo_RandomizedSearchCV()
    n_repeats: parâmetro utilizado pelo roda_modelo_RandomizedSearchCV()
    param_distributions: dicionário com parâmetros a serem testados
            ex: hiperparams = { 
                    "KNeighbors" : {
                        "n_neighbors" : randint(2, 20),
                        "weights" : ["uniform", "distance"],
                        "leaf_size" : randint(25, 100),
                        "metric" : ["minkowski","wminkowski", "euclidean"] 
                    },
                    "Gaussian" : {
                        "n_restarts_optimizer" : randint(0, 5),
                        "max_iter_predict" : randint(50,500)
                    }
                }
    n_iter: parâmetro utilizado pelo roda_modelo_RandomizedSearchCV()
    showMsg: Imprime o tempo de processamento.
             Essa informação está no dataFrame de retorno.

  Retorno
  -------
    df_retorno_rand: dataFrame com o resultado do processamento com os seguintes campos:
        [Nome, Modelo, AUC, Train AUC, Std AUC, Best Params, Tempo, objRandomizedSearchCV]

  """
  df_retorno_rand = pd.DataFrame()

  for name, model in zip(names, models):
    start_time = time.time()
    busca = roda_modelo_RandomizedSearchCV(model, dados, n_splits, n_repeats, param_distributions[name], n_iter)
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
                              resultados.iloc[busca.best_index_]['std_test_score'], busca.best_params_, int(total_time), busca]],
                            columns=['Nome', 'Modelo', 'AUC', 'Train AUC', 'Std AUC', 'Best Params', 'Tempo', 'objRandomizedSearchCV'])

    df_retorno_rand = df_retorno_rand.append(df_result2)

  df_retorno_rand.reset_index(drop=True, inplace=True)
  df_retorno_rand = df_retorno_rand.set_index('Nome')
  # df_retorno_rand = df_retorno_rand.sort_values(by="AUC", ascending=False)

  return df_retorno_rand



def plotar_curva_roc_modelos(names, models, dados):
  """
   Função para plotar a curva ROC AUC da array de names e models, sobre os dados
   sujeito a aleatoriedade do train_test_split
  """
  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  np.random.seed(73246)
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

  np.random.seed(73246)
  X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)
  model.fit(X_train, y_train)

  plot_confusion_matrix(model, X_test, y_test)



def plotar_matrix_confusao_modelos(names, models, dados, nrows, ncols):
  """
  Plotar a Matriz de Confusão do array de names e models, sobre os dados, em nrows e ncols


  Retorna um array dos modelos treinados
  """

  fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,10))
  axis = []
  for ax in axes:
    axis.extend(ax)
  axes_ind = 0

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)

  np.random.seed(73246)
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
    f1score = f1_score(y_test, y_predict)
    #axis[axes_ind].set_title(f'{name}, F1 = {f1score:.2f}', fontsize=13)
    axis[axes_ind].set_title(f'{name}', fontsize=13)
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

  np.random.seed(73246)
  X_train, X_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.3)

  dfs = []
  for name, model in zip(names, models):
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    dfs.append(format_classification_report(y_test, y_predict))

  dffinal = pd.concat(dfs, axis=1, keys=names)
  return dffinal



def plotar_media_curva_roc(names, models, dados, n_splits=5, n_repeats=10, plotar=True):
  """
  
  Função que plota a curva ROC AUC, considerando a média dos (n_splits * n_repeats) 
  de processamento com RepeatedStratifiedKFold()
  
  Parâmetros
  ----------
    names: array com o nome dos modelos
    models: array com a instancia do Modelo de Machine Learning.
    dados: dataFrame com os dados
    n_splits: quantidade de splits de dados usado pela função RepeatedStratifiedKFold
    n_repeats: quantidade de repeats usado pela função RepeatedStratifiedKFold
    plotar: plota o gráfico. Pode ser utilizado para obter a AUC média apenas.

  Retorno
  -------
    retorno: array com o nome do modelo na 1ª coluna e a AUC média na 2ª coluna
  
  """

  x_columns = dados.columns
  y = dados['ICU']
  x = dados[x_columns].drop(["ICU"], axis=1)
  retorno = []

  for name, clf in zip(names, models):
    np.random.seed(73246)
    i = 1
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,1,(n_splits * n_repeats))
    
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats)

    for train,test in cv.split(x,y):
      prediction = clf.fit(x.iloc[train],y.iloc[train]).predict_proba(x.iloc[test])
      fpr, tpr, t = metrics.roc_curve(y[test], prediction[:, 1])
      tprs.append(np.interp(mean_fpr, fpr, tpr))
      roc_auc = metrics.auc(fpr, tpr)
      aucs.append(roc_auc)
      if plotar:
        # plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='') 
      i= i+1

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)

    if plotar:
      plt.plot([0,1],[0,1],linestyle = '--', lw=2, color='red')
      plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Media ROC (AUC = %0.4f )' % (mean_auc), lw=2, alpha=1)
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



def montar_dataframe_medias_AUC(media_auc_padrao, media_auc_hiper):
  """

  Função para montar um dataFrame com os retornos da função plotar_media_curva_roc()
  
  Parâmetros
  ----------
    media_auc_padrao: array com o resultado da função plotar_media_curva_roc() sobre
    os modelos default
    media_auc_hiper: array com o resultado da função plotar_media_curva_roc() sobre
    os modelos ajustados 

  Retorno
  -------
    dfretorno: dataFrame com o merge dos parâmetros de entrada e com a coluna de diferença

  """
  
  
  def max_value(row):
    return max(row['Media_Ajustado'], row['Media_padrao'])

  dfmean1 = pd.DataFrame(data=media_auc_padrao, columns=['Nome','Media_padrao']).set_index("Nome")
  dfmean2 = pd.DataFrame(data=media_auc_hiper, columns=['Nome','Media_Ajustado']).set_index("Nome")
  dfretorno = dfmean1.merge(dfmean2,on="Nome")
  dfretorno['diferenca (%)'] = (dfretorno['Media_Ajustado'] - dfretorno['Media_padrao']) * 100
  dfretorno['AUC'] = dfretorno.apply(max_value, axis=1)
  dfretorno = dfretorno.sort_values(by='AUC', ascending=False)

  return dfretorno



def montar_dataframe_avaliacao( names_matriz, df_from_montar_classificacao, df_from_medias_AUC):
  dfretorno = pd.DataFrame()
  for name in names_matriz:
    precision_0 = pd.DataFrame(df_from_montar_classificacao[name])['precision']['0']
    precision_1 = pd.DataFrame(df_from_montar_classificacao[name])['precision']['1']
    recall_0    = pd.DataFrame(df_from_montar_classificacao[name])['recall']['0']
    recall_1    = pd.DataFrame(df_from_montar_classificacao[name])['recall']['1']
    f1score_0   = pd.DataFrame(df_from_montar_classificacao[name])['f1-score']['0']
    f1score_1   = pd.DataFrame(df_from_montar_classificacao[name])['f1-score']['1']
    accuracy    = pd.DataFrame(df_from_montar_classificacao[name])['f1-score']['accuracy']
    df_x = pd.DataFrame([[name, precision_0, precision_1, recall_0, recall_1, 
                        f1score_0, f1score_1, accuracy]], 
                      columns=['Nome','Precision 0','Precision 1','Recall 0','Recall 1',
                               'F1-Score 0', 'F1-Score 1', 'Accuracy'])
    dfretorno = dfretorno.append(df_x)

  dfretorno = dfretorno.merge(df_from_medias_AUC[['Nome','AUC']], on="Nome")
  dfretorno.reset_index(drop=True, inplace=True)
  dfretorno = dfretorno.set_index('Nome')
  

  return dfretorno


def listar_parametros(names, models):
  """
  
  Função que montar uma tabela com os parâmetros resultantes do processamento de
  otimização dos modelos, destacando o rank_test_score = 1.
  
  Parâmetros
  ----------
    names: array com o nome dos modelos
    models: array com a instancia do Modelo de Machine Learning.

  Retorno
  -------
    Nenhum retorno
  
  """

  def highlight_equal(s, value, column):
    is_max = pd.Series(data=False, index=s.index)
    is_max[column] = s.loc[column] == value
    return ['background-color: yellow; font-weight: bold;' if is_max.any() else None for v in is_max]

  for indice in names:
    obj = models.at[indice,'objRandomizedSearchCV']
    dfParams = pd.DataFrame(obj.cv_results_['params']).reset_index()
    metricas = { 'rank_test_score': obj.cv_results_['rank_test_score'],
                 'media_AUC_teste': obj.cv_results_['mean_test_score'],
                 'media_AUC_treino':obj.cv_results_['mean_train_score'] }
    dfMetricas = pd.DataFrame(metricas).reset_index()
    dfAvaliacao = pd.merge(dfMetricas,dfParams, on="index")
    dfAvaliacao.drop(['index'],axis=1,inplace=True)
    print(200*'_')
    print(indice)
    print(len(indice)*'¨')
    display(dfAvaliacao.style.apply(highlight_equal, value=1, column=['rank_test_score'], axis=1))

  print(200*'_')


