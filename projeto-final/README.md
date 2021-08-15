
# Projeto de Data Science Aplicada 2 

<p align="center">
  <img src=".\img\hospitalsiriolibanes-imagem1.jpg"/>
</p>

## Projeto Final de conclusão de Curso
Apresentação do projeto final do Bootcamp de Data Science Aplicada 2 da [Alura](https://www.alura.com.br) - Plataforma do [Bootcamp](https://bootcamps.alura.com.br/acesso-a-plataforma)


# Sumário
1. [Introdução](#intro)
2. [Dados](#dados)
3. [Objetivo](#objetivo)
4. [Metodologia](#metodologia)
5. [Referências](#referencia)


<a name="intro"></a>
# 1. Introdução

Bem vindo ao projeto final de conclusão do Bootcamp de Data Science Aplicada, segunda turma, by [Alura](http://www.alura.com.br)!

Nesse projeto trabalharemos com informações do Hospital Sírio Libanês (HSL) – São Paulo/Brasília - com o objetivo de prever quais pacientes precisarão ser admitidos na unidade de terapia intensiva (UTI) e assim, definir qual a necessidade de leitos de UTI do hospital, a partir dos dados clínicos individuais disponíveis. Definindo a quantidade de leitos necessários em um determinado hospital, é possível evitar rupturas, visto que, caso outra pessoa procure ajuda e eventualmente precise de cuidados intensivos, o modelo preditivo já conseguirá detectar essa necessidade e, desta forma, a remoção e transferência deste paciente pode ser organizada antecipadamente.


<a name="dados"></a>
# 2. Dados

As informações para o desenvolvimento desse projeto foram disponibilizadas no repositório do [Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19). Nela, encontramos diversos tipos de informações que foram separadas em 4 grupos:

* Informação demográfica - 3 variáveis
* Doenças pré-existentes - 9 variáveis
* Resultados do exame de sangue - 36 variáveis
* Sinais vitais - 6 variáveis

Serão aplicados as técnicas de obtenção, limpeza e tratamento dos dados buscando deixar as informações mais significativas para a análise, e esse tratamento pode ser visualizado no notebook [projeto_final_tratamento_dados.ipynb](https://github.com/hc0rd31r0/Bootcamp_Data_Science/blob/main/projeto-final/projeto_final_tratamento_dados.ipynb).


<a name="objetivo"></a>
# 3. Objetivo

A pandemia de covid-19 sobrecarregou o Sistema de Saúde Brasileiro, afetando principalmente a disponibilidade de leitos de UTI. É evidente que a obtenção de dados precisos é necessária para evitar colapsos e a sobrecarga dos hospitais, já que muitos estiveram com leitos de UTIs lotados. Com base nesses dados, será possível prever o que acontecerá com os próximos pacientes.

O problema proposto envolve duas tarefas (conforme descrito no site Kaggle):

**Tarefa 01**

**Prever admissão na UTI de casos confirmados de COVID-19**. Com base nos dados disponíveis, é viável prever quais pacientes precisarão de suporte em unidade de terapia intensiva? O objetivo é fornecer aos hospitais terciários e quaternários a resposta mais precisa, para que os recursos da UTI possam ser arranjados ou a transferência do paciente possa ser agendada.

**Tarefa 02**

**Prever NÃO admissão à UTI de casos COVID-19 confirmados**. Com base nos dados disponíveis, é possível prever quais pacientes precisarão de suporte de unidade de terapia intensiva? O objetivo é fornecer aos hospitais locais e temporários uma resposta boa o suficiente, para que os médicos de linha de frente possam dar alta com segurança e acompanhar remotamente esses pacientes. 


<a name="metodologia"></a>
# 4. Metodologia

Serão aplicados modelos de Machine Learning para o problema de **Classificação Binária** (a UTI é necessária? Sim ou não) proposto pelo Hospital Sírio Libanês. Formalmente o Machine Learning é definido como:

>Aprendizado de máquina é definido por um sistema computacional que busca realizar uma tarefa ***T***, aprendendo a partir de uma experiência ***E***, procurando melhorar uma performance ***P***.

Como os dados utilizados para treinar nosso modelo contém a resposta desejada, será aplicado o  **Aprendizado Supervisionado**. Os modelos mais conhecidas são Regressão Linear, Regressão Logística, Redes Neurais Artificiais, Máquina de Suporte Vetorial (ou máquinas kernel), Árvores de Decisão, K-Vizinhos mais próximos e Bayes ingênuo.

Para tal, um array com alguns modelos de Machine Learning foi utilizado e foi aplicado os conceitos apresentados nas aulas, clonando e ajustando o código python que foi encontrado nas pesquisas necessárias para a conclusão desse projeto.

## Estrutura
O projeto está organizado da seguinte forma:

* A análise principal está no arquivo [Bootcamp_DataScience_projeto_final.ipynb](https://github.com/hc0rd31r0/Bootcamp_Data_Science/blob/main/projeto-final/Bootcamp_DataScience_projeto_final.ipynb)
* O arquivo [projeto_final_hiperparametros.ipynb](https://github.com/hc0rd31r0/Bootcamp_Data_Science/blob/main/projeto-final/projeto_final_hiperparametros.ipynb) contém os teste de hiperparâmetros dos modelos e foi separado da análise principal pois requer tempo de processamento
* O arquivo [projeto_final_tratamento_dados.ipynb](https://github.com/hc0rd31r0/Bootcamp_Data_Science/blob/main/projeto-final/projeto_final_tratamento_dados.ipynb) faz o tratamento dos dados originais fornecido pelo HSL
* o arquivo [funcoes.py](https://github.com/hc0rd31r0/Bootcamp_Data_Science/blob/main/projeto-final/funcoes.py) tem o código fonte das funções utilizadas pelos 3 notebooks deixando-os mais claros.
* pasta ```dados``` contém as planilhas utilizadas, uma cópia do dataFrame resultante do processamento de otimização (arquivos: dfmodelosHP) e o nosso modelo salvo (modelo_hsl)
* pasta ```img``` contém as imagens utilizadas no projeto

<a name="referencia"></a>
# 5. Referências

* [Hospital Sírio-Libanês](https://www.hospitalsiriolibanes.org.br/Paginas/nova-home.aspx)
* [Kaggle - COVID-19 - Clinical Data to assess diagnosis - Sírio Libanês](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19)
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
* [SciKit Learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
* [Os Três Tipos de Aprendizado de Máquina](https://lamfo-unb.github.io/2017/07/27/tres-tipos-am/)
* [Enriched Lightgbm | PR 86% - notebook](https://www.kaggle.com/andrewmvd/enriched-lightgbm-pr-86-auc-92-68)
* [PedroHCAlmeida - notebook](https://github.com/PedroHCAlmeida/Bootcamp_alura/blob/main/Modulo_4/Aulas/Aulas.ipynb)
* [willianrocha (notebook)](https://github.com/willianrocha/COVID-19_clinical_data_assess_diagnosis/blob/main/notebooks/ML.ipynb)
* [Configurar o treinamento do AutoML com Python](https://docs.microsoft.com/pt-br/azure/machine-learning/how-to-configure-auto-train)
* [Avaliando os resultados do experimento de machine learning automatizado](https://docs.microsoft.com/pt-br/azure/machine-learning/how-to-understand-automated-ml)
* [Matriz de Confusão e AUC ROC](https://medium.com/data-hackers/matriz-de-confus%C3%A3o-e-auc-roc-f7e446dca107)
* [SciKit Learn - 3.3. Metrics and scoring](https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter)
* [SciKit Learn - Choosing the right estimator](https://scikit-learn.org/stable/tutorial/machine_learning_map/index.html)


----

<p align="center">
  <img src=".\img\hospitalsiriolibanes-logo3.png"/>
</p>

----