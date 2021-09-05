import streamlit as st
from joblib import load
import pandas as pd
from utils import Transformador

st.image('https://raw.githubusercontent.com/hc0rd31r0/Bootcamp_Data_Science/main/avaliacao_credito/img/bytebank_logo.png')
st.write('# Simulador de Crédito')

#Cor de fundo do listbox
st.markdown('<style>div[role="listbox"] ul{background-color: #eee1f79e};</style>', unsafe_allow_html=True)
st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)


def avaliar_mau(dict_respostas):
	modelo = load('./objetos/modelo.joblib')
	features = load('./objetos/features.joblib')

	if dict_respostas['Anos_desempregado'] > 0:
		dict_respostas['Anos_empregado'] = dict_respostas['Anos_desempregado'] * -1 
	
	respostas = []
	for coluna in features:
		respostas.append(dict_respostas[coluna])

	df_novo_cliente = pd.DataFrame(data=[respostas], columns=features)

	mau = modelo.predict(df_novo_cliente)
	print(mau)
	return mau[0]


my_expander_1 = st.beta_expander("Trabalho")

my_expander_2 = st.beta_expander("Pessoal")

my_expander_3 = st.beta_expander("Familia")

dict_respostas = {}
lista_campos = load('objetos/lista_campos.joblib')

with my_expander_1:

	col1_form, col2_form = st.beta_columns(2)

	dict_respostas['Categoria_de_renda'] = col1_form.selectbox('Qual a categoria de renda ?', lista_campos['Categoria_de_renda'])

	dict_respostas['Ocupacao'] = col1_form.selectbox('Qual a Ocupação ?', lista_campos['Ocupacao'])

	# dict_respostas['Tem_telefone_trabalho'] = 1 if col1_form.selectbox('Tem um telefone do trabalho ?', ['Sim', 'Não']) == 'Sim' else 0
	dict_respostas['Tem_telefone_trabalho'] = 1  if col1_form.radio( label='Tem um telefone do trabalho ?', options=('Sim', 'Não'),  index=0 ) == 'Sim' else 0

	dict_respostas['Rendimento_Anual'] = col2_form.slider('Qual o salário mensal ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=35000, step=500) * 13

	dict_respostas['Anos_empregado'] = col2_form.slider('Quantos anos empregado ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)

	dict_respostas['Anos_desempregado'] = col2_form.slider('Quantos anos desempregado ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=50, step=1)

with my_expander_2:

    col3_form, col4_form = st.beta_columns(2)

    dict_respostas['Grau_Escolaridade'] = col3_form.selectbox('Qual o Grau de Escolaridade ?', lista_campos['Grau_Escolaridade'])

    dict_respostas['Estado_Civil'] = col3_form.selectbox('Qual o Estado Civil ?', lista_campos['Estado_Civil'])

    #dict_respostas['Tem_Carro'] = 1 if col3_form.selectbox('Tem um Carro ?', ['Sim', 'Não']) == 'Sim' else 0
    dict_respostas['Tem_Carro'] = 1  if col3_form.radio('Tem um Carro ?', ['Sim', 'Não'],  index=0 ) == 'Sim' else 0

    #dict_respostas['Tem_telefone_fixo'] = 1 if col4_form.selectbox('Tem um telefone fixo ?', ['Sim', 'Não']) == 'Sim' else 0
    dict_respostas['Tem_telefone_fixo'] = 1 if col4_form.radio('Tem um telefone fixo ?', ['Sim', 'Não'], index=0) == 'Sim' else 0

    #dict_respostas['Tem_email'] = 1 if col4_form.selectbox('Tem um email ?', ['Sim', 'Não']) == 'Sim' else 0
    dict_respostas['Tem_email'] = 1 if col4_form.radio('Tem um email ?', ['Sim', 'Não'], index=0) == 'Sim' else 0

    dict_respostas['Idade'] = col4_form.slider('Qual a idade ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=100, step=1)

with my_expander_3:

    col4_form, col5_form = st.beta_columns(2)

    dict_respostas['Moradia'] = col4_form.selectbox('Qual o tipo de moradia ?', lista_campos['Moradia'])

    #dict_respostas['Tem_Casa_Propria'] = 1 if col4_form.selectbox('Tem Casa Propria ?', ['Sim', 'Não']) == 'Sim' else 0
    dict_respostas['Tem_Casa_Propria'] = 1 if col4_form.radio('Tem Casa Propria ?', ['Sim', 'Não'], index=0) == 'Sim' else 0

    dict_respostas['Tamanho_Familia'] = col5_form.slider('Qual o tamanho da familia ?', help='Podemos mover a barra usando as setas do teclado', min_value=1, max_value=20, step=1)

    dict_respostas['Qtd_Filhos'] = col5_form.slider('Quantos filhos ?', help='Podemos mover a barra usando as setas do teclado', min_value=0, max_value=20, step=1)


if st.button('Avaliar crédito'):
	if avaliar_mau(dict_respostas):
		st.error('Crédito negado')
	else:
		st.success('Crédito Aprovado')