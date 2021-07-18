
# Projeto de Data Science Aplicada 2 

<p align="center">
  <img src=".\img\hospitalsiriolibanes-imagem1.jpg"/>
</p>

## Projeto Final de conclusão de Curso


# Sumário
1. [Introdução](#intro)
2. [Dados](#dados)
3. [Análises](#analise)
4. [Referências](#referencia)


<a name="intro"></a>
# Introdução

Bem vindo ao projeto final de conclusão do Bootcamp de Data Science Aplicada, segunda turma, by [Alura](http://www.alura.com.br)!

## Objetivo
Nesse projeto vamos trabalhar com informações do Hospital Sírio Libanês - São Paulo e Brasília - e o objetivo será **prever** quais pacientes precisarão ser admitidos na unidade de terapia intensiva e assim, **definir** qual a necessidade de leitos de UTI do hospital, a partir dos dados clínicos individuais disponíveis. Quando conseguimos definir a quantidade de leitos necessários em um determinado hospital, conseguimos evitar rupturas, visto que, caso outra pessoa procure ajuda e, eventualmente, precise de cuidados intensivos, o modelo preditivo já conseguirá detectar essa necessidade e, desta forma, a remoção e transferência deste(a) paciente pode ser organizada antecipadamente.

Para atingir esse objetivo buscarei aplicar todo conhecimento adquirido durante o curso de Data Science e na experiência que venho adquirindo desde meu curso Técnico em Computação que fiz lá no 2º Grau, hoje ensino médio. São mais de 30 anos estudando e 25 anos trabalhando com Tecnologia da Informação.


<a name="dados"></a>
# Dados

As informações para o desenvolvimento desse projeto foram disponibilizadas no repositório do [Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19). Nela, encontramos diversos tipos de informações que foram separadas em 4 grupos:

* Informação demográfica - 3 variáveis
* Doenças pré-existentes - 9 variáveis
* Resultados do exame de sangue - 36 variáveis
* Sinais vitais - 6 variáveis

Serão aplicados as técnicas de obtenção, limpeza e tratamento dos dados buscando deixar as informações mais significativas para a análise.


<a name="analise"></a>
# Análises

A pandemia de COVID-19 causou uma sobrecarga nos sistemas de saúde, que afetou a disponibilidade de leitos de UTI nos hospitais. Portanto, é de extrema importância a obtenção de dados precisos para prever e preparar os sistemas de saúde e evitar colapsos, definidos pela necessidade de leitos de UTI acima da capacidade. Essas previsões podem ser realizadas utilizando os dados clínicos individuais.

O problema proposto envolve duas tarefas (conforme descrito no site Kaggle):

**Tarefa 01**

**Prever admissão na UTI de casos confirmados de COVID-19**. Com base nos dados disponíveis, é viável prever quais pacientes precisarão de suporte em unidade de terapia intensiva? O objetivo é fornecer aos hospitais terciários e quaternários a resposta mais precisa, para que os recursos da UTI possam ser arranjados ou a transferência do paciente possa ser agendada.

**Tarefa 02**

**Prever NÃO admissão à UTI de casos COVID-19 confirmados**. Com base nos dados disponíveis, é possível prever quais pacientes precisarão de suporte de unidade de terapia intensiva? O objetivo é fornecer aos hospitais locais e temporários uma resposta boa o suficiente, para que os médicos de linha de frente possam dar alta com segurança e acompanhar remotamente esses pacientes. 


Serão aplicados modelos de Machine Learning para o problema de <u>Classificação Binária</u> (vai necessitar UTI, ou não) proposto pelo Hospital Sírio Libanês.
Formalmente o Machine Learning é definido como:

>Aprendizado de máquina é definido por um sistema computacional que busca realizar uma tarefa $T$, aprendendo a partir de uma experiência $E$, procurando melhorar uma performance $P$.

Considerando que os dados que utilizaremos para treinar nosso modelo contém a resposta desejada, usaremos técnicas para resolver problemas de **Aprendizado Supervisionado**, entre as mais conhecidas estão regressão linear, regressão logística, redes neurais artificiais, máquina se suporte vetorial (ou máquinas kernel), árvores de decisão, k-vizinhos mais próximos e Bayes ingênuo. Como ainda não sou especialista no assunto, utilizei:   [[[ Apresentar as técnicas utilizadas ]]]


<a name="referencia"></a>
# Referências

* [Kaggle](https://www.kaggle.com/S%C3%ADrio-Libanes/covid19)
* [Pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)
* [SciKit Learn](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model)
* [Os Três Tipos de Aprendizado de Máquina](https://lamfo-unb.github.io/2017/07/27/tres-tipos-am/)
* [Enriched Lightgbm | PR 86% notebook](https://www.kaggle.com/andrewmvd/enriched-lightgbm-pr-86-auc-92-68) 
