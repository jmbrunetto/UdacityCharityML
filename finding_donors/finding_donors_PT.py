#!/usr/bin/env python
# coding: utf-8

# # Nanodegree Engenheiro de Machine Learning
# ## Aprendizado Supervisionado
# ## Projeto: Encontrando doadores para a *CharityML*

# Seja bem-vindo ao segundo projeto do Nanodegree Engenheiro de Machine Learning! Neste notebook, você receberá alguns códigos de exemplo e será seu trabalho implementar as funcionalidades adicionais necessárias para a conclusão do projeto. As seções cujo cabeçalho começa com **'Implementação'** indicam que o bloco de código posterior requer funcionalidades adicionais que você deve desenvolver. Para cada parte do projeto serão fornecidas instruções e as diretrizes da implementação estarão marcadas no bloco de código com uma expressão `'TODO'`. 
# Por favor, leia cuidadosamente as instruções!
# 
# Além de implementações de código, você terá de responder questões relacionadas ao projeto e à sua implementação. Cada seção onde você responderá uma questão terá um cabeçalho com o termo **'Questão X'**. Leia com atenção as questões e forneça respostas completas nas caixas de texto que começam com o termo **'Resposta:'**. A submissão do seu projeto será avaliada baseada nas suas resostas para cada uma das questões além das implementações que você disponibilizar.
# 
# >**Nota:** Por favor, especifique QUAL A VERSÃO DO PYTHON utilizada por você para a submissão deste notebook. As células "Code" e "Markdown" podem ser executadas utilizando o atalho do teclado **Shift + Enter**. Além disso, as células "Markdown" podem ser editadas clicando-se duas vezes na célula.
# 

# ## Iniciando
# 
# Neste projeto, você utilizará diversos algoritmos de aprendizado supervisionado para modelar com precisão a remuneração de indivíduos utilizando dados coletados no censo americano de 1994. Você escolherá o algoritmo mais adequado através dos resultados preliminares e irá otimizá-lo para modelagem dos dados. O seu objetivo com esta implementação é construir um modelo que pode predizer com precisão se um indivíduo possui uma remuneração superior a $50,000. Este tipo de tarefa pode surgir em organizações sem fins lucrativos que sobrevivem de doações. Entender a remuneração de um indivíduo pode ajudar a organização o montante mais adequado para uma solicitação de doação, ou ainda se eles realmente deveriam entrar em contato com a pessoa. Enquanto pode ser uma tarefa difícil determinar a faixa de renda de uma pesssoa de maneira direta, nós podemos inferir estes valores através de outros recursos disponíveis publicamente. 
# 
# O conjunto de dados para este projeto se origina do [Repositório de Machine Learning UCI](https://archive.ics.uci.edu/ml/datasets/Census+Income) e foi cedido por Ron Kohavi e Barry Becker, após a sua publicação no artigo _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. Você pode encontrar o artigo de Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). Os dados que investigaremos aqui possuem algumas pequenas modificações se comparados com os dados originais, como por exemplo a remoção da funcionalidade `'fnlwgt'` e a remoção de registros inconsistentes.
# 

# ----
# ## Explorando os dados
# Execute a célula de código abaixo para carregas as bibliotecas Python necessárias e carregas os dados do censo. Perceba que a última coluna deste conjunto de dados, `'income'`, será o rótulo do nosso alvo (se um indivíduo possui remuneração igual ou maior do que $50,000 anualmente). Todas as outras colunas são dados de cada indívduo na base de dados do censo.

# In[2]:


# Importe as bibliotecas necessárias para o projeto.
import numpy as np
import pandas as pd
from time import time
# from IPython.display import display # Permite a utilização da função display() para DataFrames.

# Importação da biblioteca de visualização visuals.py
import visuals as vs

# Exibição amigável para notebooks
# get_ipython().run_line_magic('matplotlib', 'inline')

# Carregando os dados do Censo
data = pd.read_csv("census.csv")

# Sucesso - Exibindo o primeiro registro
print(data.head(n=1))


# ### Implementação: Explorando os Dados
# 
# Uma investigação superficial da massa de dados determinará quantos indivíduos se enquadram em cada grupo e nos dirá sobre o percentual destes indivúdos com remuneração anual superior à \$50,000. No código abaixo, você precisará calcular o seguinte:
# - O número total de registros, `'n_records'`
# - O número de indivíduos com remuneração anual superior à \$50,000, `'n_greater_50k'`.
# - O número de indivíduos com remuneração anual até \$50,000, `'n_at_most_50k'`.
# - O percentual de indivíduos com remuneração anual superior à \$50,000, `'greater_percent'`.
# 
# ** DICA: ** Você pode precisar olhar a tabela acima para entender como os registros da coluna `'income'` estão formatados.

# In[3]:


# TODO: Número total de registros.
n_records = data.shape[0]

# TODO: Número de registros com remuneração anual superior à $50,000
n_greater_50k = data[data.income == '>50K'].shape[0]

# TODO: O número de registros com remuneração anual até $50,000
n_at_most_50k = data[data.income == '<=50K'].shape[0]

# TODO: O percentual de indivíduos com remuneração anual superior à $50,000
greater_percent = (n_greater_50k/n_records)*100

# Exibindo os resultados
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))


# ** Explorando as colunas **
# * **age**: contínuo. 
# * **workclass**: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov, Local-gov, State-gov, Without-pay, Never-worked. 
# * **education**: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool. 
# * **education-num**: contínuo. 
# * **marital-status**: Married-civ-spouse, Divorced, Never-married, Separated, Widowed, Married-spouse-absent, Married-AF-spouse. 
# * **occupation**: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct, Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv, Protective-serv, Armed-Forces. 
# * **relationship**: Wife, Own-child, Husband, Not-in-family, Other-relative, Unmarried. 
# * **race**: Black, White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other. 
# * **sex**: Female, Male. 
# * **capital-gain**: contínuo. 
# * **capital-loss**: contínuo. 
# * **hours-per-week**: contínuo. 
# * **native-country**: United-States, Cambodia, England, Puerto-Rico, Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan, Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy, Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France, Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia, Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.

# ----
# ## Preparando os dados
# Antes de que os dados possam ser utilizados como input para algoritmos de machine learning, muitas vezes eles precisam ser tratados, formatados e reestruturados — este processo é conhecido como **pré-processamento**. Felizmente neste conjunto de dados não existem registros inconsistentes para tratamento, porém algumas colunas precisam ser ajustadas. Este pré-processamento pode ajudar muito com o resultado e poder de predição de quase todos os algoritmos de aprendizado.

# ### Transformando os principais desvios das colunas contínuas
# Um conjunto de dados pode conter ao menos uma coluna onde os valores tendem a se próximar para um único número, mas também podem conter registros com o mesmo atributo contendo um valor muito maior ou muito menor do que esta tendência. Algoritmos podem ser sensíveis para estes casos de distribuição de valores e este fator pode prejudicar sua performance se a distribuição não estiver normalizada de maneira adequada. Com o conjunto de dados do censo, dois atributos se encaixam nesta descrição: '`capital-gain'` e `'capital-loss'`.
# 
# Execute o código da célula abaixo para plotar um histograma destes dois atributos. Repare na distribuição destes valores.

# In[4]:


# Dividindo os dados entre features e coluna alvo
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# Visualizando os principais desvios das colunas contínuas entre os dados
# vs.distribution(data)


# Para atributos com distribuição muito distorcida, tais como `'capital-gain'` e `'capital-loss'`, é uma prática comum aplicar uma <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">transformação logarítmica</a> nos dados para que os valores muito grandes e muito pequenos não afetem a performance do algoritmo de aprendizado. Usar a transformação logarítmica reduz significativamente os limites dos valores afetados pelos outliers (valores muito grandes ou muito pequenos). Deve-se tomar cuidado ao aplicar esta transformação, poir o logaritmo de `0` é indefinido, portanto temos que incrementar os valores em uma pequena quantia acima de `0` para aplicar o logaritmo adequadamente.
# 
# Execute o código da célula abaixo para realizar a transformação nos dados e visualizar os resultados. De novo, note os valores limite e como os valores estão distribuídos.

# In[5]:


# Aplicando a transformação de log nos registros distorcidos.
skewed = ['capital-gain', 'capital-loss']
features_log_transformed = pd.DataFrame(data = features_raw)
features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

# Visualizando as novas distribuições após a transformação.
# vs.distribution(features_log_transformed, transformed = True)


# ### Normalizando atributos numéricos
# Além das transformações em atributos distorcidos, é uma boa prática comum realizar algum tipo de adaptação de escala nos atributos numéricos. Ajustar a escala nos dados não modifica o formato da distribuição de cada coluna (tais como `'capital-gain'` ou `'capital-loss'` acima); no entanto, a normalização garante que cada atributo será tratado com o mesmo peso durante a aplicação de aprendizado supervisionado. Note que uma vez aplicada a escala, a observação dos dados não terá o significado original, como exemplificado abaixo.
# 
# Execute o código da célula abaixo para normalizar cada atributo numérico, nós usaremos ara isso a [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html).

# In[6]:


# Importando sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Inicializando um aplicador de escala e aplicando em seguida aos atributos
scaler = MinMaxScaler() # default=(0, 1)
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

# Exibindo um exemplo de registro com a escala aplicada
print(features_log_minmax_transform.head(n=5))


# ### Implementação: Pré-processamento dos dados
# 
# A partir da tabela em **Explorando os dados** acima, nós podemos observar que existem diversos atributos não-numéricos para cada registro. Usualmente, algoritmos de aprendizado esperam que os inputs sejam numéricos, o que requer que os atributos não numéricos (chamados de *variáveis de categoria*) sejam convertidos. Uma maneira popular de converter as variáveis de categoria é utilizar a estratégia **one-hot encoding**. Esta estratégia cria uma variável para cada categoria possível de cada atributo não numérico. Por exemplo, assuma que `algumAtributo` possuí três valores possíveis: `A`, `B`, ou `C`. Nós então transformamos este atributo em três novos atributos: `algumAtributo_A`, `algumAtributo_B` e `algumAtributo_C`.
# 
# 
# |   | algumAtributo |                    | algumAtributo_A | algumAtributo_B | algumAtributo_C |
# | :-: | :-: |                            | :-: | :-: | :-: |
# | 0 |  B  |  | 0 | 1 | 0 |
# | 1 |  C  | ----> one-hot encode ----> | 0 | 0 | 1 |
# | 2 |  A  |  | 1 | 0 | 0 |
# 
# Além disso, assim como os atributos não-numéricos, precisaremos converter a coluna alvo não-numérica, `'income'`, para valores numéricos para que o algoritmo de aprendizado funcione. Uma vez que só existem duas categorias possíveis para esta coluna ("<=50K" e ">50K"), nós podemos evitar a utilização do one-hot encoding e simplesmente transformar estas duas categorias para `0` e `1`, respectivamente. No trecho de código abaixo, você precisará implementar o seguinte:
#  - Utilizar [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) para realizar o one-hot encoding nos dados da `'features_log_minmax_transform'`.
#  - Converter a coluna alvo `'income_raw'` para re.
#    - Transforme os registros com "<=50K" para `0` e os registros com ">50K" para `1`.

# In[7]:


# TODO: Utilize o one-hot encoding nos dados em 'features_log_minmax_transform' utilizando pandas.get_dummies()
features_final = pd.get_dummies(features_log_minmax_transform)

# TODO: Faça o encode da coluna 'income_raw' para valores numéricos
income = income_raw.replace('>50K', 1).replace('<=50K', 0)

# Exiba o número de colunas depois do one-hot encoding
encoded = list(features_final.columns)
print("{} total features after one-hot encoding.".format(len(encoded)))

# Descomente a linha abaixo para ver as colunas após o encode
print(encoded)


# ### Embaralhar e dividir os dados
# Agora todas as _variáveis de categoria_ foram convertidas em atributos numéricos e todos os atributos numéricos foram normalizados. Como sempre, nós agora dividiremos os dados entre conjuntos de treinamento e de teste. 80% dos dados serão utilizados para treinamento e 20% para teste.
# 
# Execute o código da célula abaixo para realizar divisão.

# In[8]:


# Importar train_test_split
from sklearn.model_selection import train_test_split

# Dividir os 'atributos' e 'income' entre conjuntos de treinamento e de testes.
X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))


# ----
# ## Avaliando a performance do modelo
# Nesta seção nós investigaremos quatro algoritmos diferentes e determinaremos qual deles é melhor para a modelagem dos dados. Três destes algoritmos serão algoritmos de aprendizado supervisionado de sua escolha e o quarto algoritmo é conhecido como *naive predictor*.

# ### Métricas e o Naive predictor
# 
# *CharityML*, equpada com sua pesquisa, sabe que os indivíduos que fazem mais do que \$50,000 possuem maior probabilidade de doar para a sua campanha de caridade. Por conta disto, a *CharityML* está particularmente interessada em predizer com acurácia quais indivíduos possuem remuneração acima de \$50,000. Parece que utilizar **acurácia (accuracy)** como uma métrica para avaliar a performance de um modelo é um parâmetro adequado. Além disso, identificar alguém que *não possui* remuneração acima de \$50,000 como alguém que recebe acima deste valor seria ruim para a *CharityML*, uma vez que eles estão procurando por indivíduos que desejam doar. Com isso, a habilidade do modelo em predizer com preisão aqueles que possuem a remuneração acima dos \$50,000 é *mais importante* do que a habilidade de realizar o **recall** destes indivíduos. Nós podemos utilizar a fórmula **F-beta score** como uma métrica que considera ambos: precision e recall.
# 
# 
# $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
# 
# Em particular, quando $\beta = 0.5$, maior ênfase é atribuída para a variável precision. Isso é chamado de **F$_{0.5}$ score** (ou F-score, simplificando).
# 
# Analisando a distribuição de classes (aqueles que possuem remuneração até \$50,000 e aqueles que possuem remuneração superior), fica claro que a maioria dos indivíduos não possui remuneração acima de \$50,000. Isto pode ter grande impacto na **acurácia (accuracy)**, uma vez que nós poderíamos simplesmente dizer *"Esta pessoa não possui remuneração acima de \$50,000"* e estar certos em boa parte das vezes, sem ao menos olhar os dados! Fazer este tipo de afirmação seria chamado de **naive**, uma vez que não consideramos nenhuma informação para balisar este argumento. É sempre importante considerar a *naive prediction* para seu conjunto de dados, para ajudar a estabelecer um benchmark para análise da performance dos modelos. Com isso, sabemos que utilizar a naive prediction não traria resultado algum: Se a predição apontasse que todas as pessoas possuem remuneração inferior à \$50,000, a *CharityML* não identificaria ninguém como potencial doador. 
# 
# 
# 
# #### Nota: Revisando: accuracy, precision e recall
# 
# ** Accuracy ** mede com que frequência o classificador faz a predição correta. É a proporção entre o número de predições corretas e o número total de predições (o número de registros testados).
# 
# ** Precision ** informa qual a proporção de mensagens classificamos como spam eram realmente spam. Ou seja, é a proporção de verdadeiros positivos (mensagens classificadas como spam que eram realmente spam) sobre todos os positivos (todas as palavras classificadas como spam, independente se a classificação estava correta), em outras palavras, é a proporção
# 
# `[Verdadeiros positivos/(Verdadeiros positivos + Falso positivos)]`
# 
# ** Recall(sensibilidade)** nos informa qual a proporção das mensagens que eram spam que foram corretamente classificadas como spam. É a proporção entre os verdadeiros positivos (classificados como spam, que realmente eram spam) sobre todas as palavras que realmente eram spam. Em outras palavras, é a proporção entre
# 
# `[Verdadeiros positivos/(Verdadeiros positivos + Falso negativos)]`
# 
# Para problemas de classificação distorcidos em suas distribuições, como no nosso caso, por exemplo, se tivéssemos 100 mensagems de texto e apenas 2 fossem spam e todas as outras não fossem, a "accuracy" por si só não seria uma métrica tão boa. Nós poderiamos classificar 90 mensagems como "não-spam" (incluindo as 2 que eram spam mas que teriam sido classificadas como não-spam e, por tanto, seriam falso negativas.) e 10 mensagems como spam (todas as 10 falso positivas) e ainda assim teriamos uma boa pontuação de accuracy. Para estess casos, precision e recall são muito úteis. Estas duas métricas podem ser combinadas para resgatar o F1 score, que é calculado através da média(harmônica) dos valores de precision e de recall. Este score pode variar entre 0 e 1, sendo 1 o melhor resultado possível para o F1 score (consideramos a média harmônica pois estamos lidando com proporções).

# ### Questão 1 - Performance do Naive Predictor
# * Se escolhessemos um modelo que sempre prediz que um indivíduo possui remuneração acima de $50,000, qual seria a accuracy e o F-score considerando este conjunto de dados? Você deverá utilizar o código da célula abaixo e atribuir os seus resultados para as variáveis `'accuracy'` e `'fscore'` que serão usadas posteriormente.
# 
# ** Por favor, note ** que o propósito ao gerar um naive predictor é simplesmente exibir como um modelo sem nenhuma inteligência se comportaria. No mundo real, idealmente o seu modelo de base será o resultado de um modelo anterior ou poderia ser baseado em um paper no qual você se basearia para melhorar. Quando não houver qualquer benchmark de modelo, utilizar um naive predictor será melhor do que uma escolha aleatória.
# 
# ** DICA: ** 
# 
# * Quando temos um modelo que sempre prediz '1' (e.x o indivíduo possui remuneração superior à 50k) então nosso modelo não terá Verdadeiros Negativos ou Falso Negativos, pois nós não estaremos afirmando que qualquer dos valores é negativo (ou '0') durante a predição. Com isso, nossa accuracy neste caso se torna o mesmo valor da precision (Verdadeiros positivos/ (Verdadeiros positivos + Falso positivos)) pois cada predição que fizemos com o valor '1' que deveria ter o valor '0' se torna um falso positivo; nosso denominador neste caso é o número total de registros.
# * Nossa pontuação de Recall(Verdadeiros positivos/(Verdadeiros Positivos + Falsos negativos)) será 1 pois não teremos Falsos negativos.

# In[16]:


'''
TP = np.sum(income) # Contando pois este é o caso "naive". Note que 'income' são os dados 'income_raw' convertidos
para valores numéricos durante o passo de pré-processamento de dados.
FP = income.count() - TP # Específico para o caso naive

TN = 0 # Sem predições negativas para o caso naive
FN = 0 # Sem predições negativas para o caso naive
'''
TP = np.sum(income) # Numero de Predições Corretas
FP = income.count() - TP # Numero de total de Predições - Predições Corretas = Falsos Positivos

# TODO: Calcular accuracy, precision e recall
accuracy = TP / income.count()
recall = 1 # Não temos Falsos Negativos então seria TP/TP
precision = TP / (TP + FP)

beta = 0.5**2

# TODO: Calcular o F-score utilizando a fórmula acima para o beta = 0.5 e os valores corretos de precision e recall.
fscore = (1+beta)*((precision*recall)/((beta*precision)+recall))

#𝐹𝛽=(1+𝛽2)⋅𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛⋅𝑟𝑒𝑐𝑎𝑙𝑙(𝛽2⋅𝑝𝑟𝑒𝑐𝑖𝑠𝑖𝑜𝑛)+𝑟𝑒𝑐𝑎𝑙𝑙

# Exibir os resultados 
print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(accuracy, fscore))


# ###  Modelos de Aprendizado Supervisionado
# **Estes são alguns dos modelos de aprendizado supervisionado disponíveis em** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html)
# - Gaussian Naive Bayes (GaussianNB)
# - Decision Trees (Árvores de decisão)
# - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
# - K-Nearest Neighbors (KNeighbors)
# - Stochastic Gradient Descent Classifier (SGDC)
# - Support Vector Machines (SVM)
# - Logistic Regression

# ### Questão 2 - Aplicação do Modelo
# Liste três dos modelos de aprendizado supervisionado acima que são apropriados para este problema que você irá testar nos dados do censo. Para cada modelo escolhido
# 
# - Descreva uma situação do mundo real onde este modelo pode ser utilizado. 
# - Quais são as vantagems da utilização deste modelo; quando ele performa bem?
# - Quais são as fraquesas do modelo; quando ele performa mal?
# - O que torna este modelo um bom candidato para o problema, considerando o que você sabe sobre o conjunto de dados?
# 
# ** DICA: **
# 
# Estruture sua resposta no mesmo formato acima^, com 4 partes para cada um dos modelos que você escolher. Por favor, inclua referências em cada uma das respostas.

# ### Gaussian Naive Bayes 
# - **GaussianNB**
# ###### Descreva uma situação do mundo real onde este modelo pode ser utilizado.
# - Geralmente utilizado para tarefas de classificação envolvendo textos, frases e sentenças. Alguns exemplos disto são classificação de emails como spam comparando palavras comumente utilizadas em spam com o conteúdo do email. Tambem pode ser utilizado para classificar categorias, baseado nos atributos como este exemplo de Sex Classification, baseado na altura/peso/tamanho do pé (https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Sex_classification).
# - [Fonte](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)
# ###### Quais são as vantagems da utilização deste modelo; quando ele performa bem?
# - Simplicidade de implementação.
# - Alta competividade.
# ###### Quais são as fraquesas do modelo; quando ele performa mal?
# - Quando os atributos não são relacionados entre si, não tem um bom resultado.
# - Com poucos dados de treino, pode se gerar um bias muito alto.
# ###### O que torna este modelo um bom candidato para o problema, considerando o que você sabe sobre o conjunto de dados?
# - Acredito que como os dados estão bem relacionados entre si, e que os doadores propensos a doar tem o mesmo padrão de comportamento espero que este algoritimo terá uma boa performance.
# 
# ### Decision Trees 
# - **Árvores de decisão**
# ###### Descreva uma situação do mundo real onde este modelo pode ser utilizado.
# - Muitas são as situações em que Decision Trees são usadas. Podemos citar o campo médico¹, onde foi utilizada para determinar a probabilidade de uma doença ocorrer, tambem é utilizada para reconhecimento de objetos em imagens² e até mesmo para prever quais ações são saudaveis para compra³.
# - ¹ - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4466856/
# - ² - http://dimatura.net/publications/dtlbp_maturana_soto_accv10.pdf
# - ³ - https://www.quantinsti.com/blog/use-decision-trees-machine-learning-predict-stock-movements		
# ###### Quais são as vantagems da utilização deste modelo; quando ele performa bem?
# - Relativamente simples de entender e implementar.
# - Não requer grandes preparações de dados, lida bem com dados numericos e categoricos.
# ###### Quais são as fraquesas do modelo; quando ele performa mal?
# - Decision Trees complexas não generalizam muito bem, causando overfitting.
# - Sensivel aos dados de input, uma pequena variação pode resultar em um resultado muito diferente.
# - Pode criar arvores com um viés errado, se uma classe dominar o dataset.
# ###### O que torna este modelo um bom candidato para o problema, considerando o que você sabe sobre o conjunto de dados?
# - Por lidar com dados categoricos e numericos, caso do nosso dataset, e por ter uma facil implementação e interpretação suponho que será um bom modelo para que eu possa analisar, e entender a lógica.
# 	
# ### Support Vector Machines
# - **SVM**
# ###### Descreva uma situação do mundo real onde este modelo pode ser utilizado.
# - Encontramos alguns exemplos de aplicação em trabalhos com imagem como Reconhecimento de Faces¹ e Reconhecimento de Escrita².
# - ¹ - https://papers.nips.cc/paper/1609-support-vector-machines-applied-to-face-recognition.pdf
# - ² - https://thesai.org/Downloads/Volume6No11/Paper_9-Handwriting_Word_Recognition_Based_on_SVM_Classifier.pdf
# ###### Quais são as vantagems da utilização deste modelo; quando ele performa bem?
# - Tem uma performance muito boa para datasets com muitas dimensões.
# - Trabalha bem com dados com ampla margem de separação.
# ###### Quais são as fraquesas do modelo; quando ele performa mal?
# - Quando temos um dataset muito largo, a performance é impactada.
# - Quando os dados se sobrepõe muito, não tem bom resultado.
# ###### O que torna este modelo um bom candidato para o problema, considerando o que você sabe sobre o conjunto de dados?
# - Por ser um modelo que trabalha bem com "High Dimentions", e ter bom desempenho em classificação binaria (nosso caso) acredito que terá uma vantagem muito boa.

# ### Implementação - Criando um Pipeline de Treinamento e Predição
# Para avaliar adequadamente a performance de cada um dos modelos que você escolheu é importante que você crie um pipeline de treinamento e predição que te permite de maneira rápida e eficiente treinar os modelos utilizando vários tamanhos de conjuntos de dados para treinamento, além de performar predições nos dados de teste. Sua implementação aqui será utilizada na próxima seção. No bloco de código abaixo, você precisará implementar o seguinte:
#  - Importar `fbeta_score` e `accuracy_score` de [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Adapte o algoritmo para os dados de treinamento e registre o tempo de treinamento. 
#  - Realize predições nos dados de teste `X_test`, e também nos 300 primeiros pontos de treinamento `X_train[:300]`.
#    - Registre o tempo total de predição. 
#  - Calcule a acurácia tanto para o conjundo de dados de treino quanto para o conjunto de testes.
#  - Calcule o F-score para os dois conjuntos de dados: treino e testes. 
#    - Garanta que você configurou o parâmetro `beta`! 

# In[ ]:


# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import fbeta_score, accuracy_score
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):

    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    results = {}
    
    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train[:sample_size], y_test[:sample_size])
    end = time() # Get end time
    
    # TODO: Calculate the training time
    results['train_time'] = end - start
        
    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = learner.accuracy_score(y_train[:300], predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = learner.accuracy_score(y_test, predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = learner.fbeta_score(y_train[:300], predictions_train, 0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = learner.fbeta_score(y_test, predictions_test, 0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results


# ### Implementação: Validação inicial do modelo
# No código da célular, você precisará implementar o seguinte:
# - Importar os três modelos de aprendizado supervisionado que você escolheu na seção anterior 
# - Inicializar os três modelos e armazená-los em `'clf_A'`, `'clf_B'`, e `'clf_C'`. 
#   - Utilize um `'random_state'` para cada modelo que você utilizar, caso seja fornecido.
#   - **Nota:** Utilize as configurações padrão para cada modelo - você otimizará um modelo específico em uma seção posterior
# - Calcule o número de registros equivalentes à 1%, 10%, e 100% dos dados de treinamento.
#   - Armazene estes valores em `'samples_1'`, `'samples_10'`, e `'samples_100'` respectivamente.
# 
# **Nota:** Dependendo do algoritmo de sua escolha, a implementação abaixo pode demorar algum tempo para executar!

# In[ ]:


# TODO: Importe os três modelos de aprendizado supervisionado da sklearn
#Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
#Decision Trees
from sklearn.tree import DecisionTreeClassifier
#Support Vector Machines
from sklearn.svm import SVC

# TODO: Inicialize os três modelos
clf_A = GaussianNB()
clf_B = DecisionTreeClassifier(random_state=0)
clf_C = SVC(random_state=0, gamma="auto")


# TODO: Calcule o número de amostras para 1%, 10%, e 100% dos dados de treinamento
# HINT: samples_100 é todo o conjunto de treinamento e.x.: len(y_train)
# HINT: samples_10 é 10% de samples_100
# HINT: samples_1 é 1% de samples_100
samples_100 = int(len(X_train))
samples_10 = int(len(X_train) / 10)
samples_1 = int(len(X_train) / 100)

# Colete os resultados dos algoritmos de aprendizado
results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        print("Testing {} for {} samples".format(clf, samples))
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

# Run metrics visualization for the three supervised learning models chosen
vs.evaluate(results, accuracy, fscore)


# TODO: Importar 'GridSearchCV', 'make_scorer', e qualquer biblioteca necessária
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, fbeta_score

# TODO: Inicializar o classificador
clf = DecisionTreeClassifier(random_state=0)

# TODO: Criar a lista de parâmetros que você quer otimizar, utilizando um dicionário, caso necessário.
# HINT: parameters = {'parameter_1': [value1, value2], 'parameter_2': [value1, value2]}
parameters = {'criterion': ['gini', 'entropy'],
              'splitter': ['best', 'random'],
              'max_depth': [2, 3, 4, 5, 10, 25, 50],
              'min_samples_split': [2, 3, 4, 8, 16, 32]}

# TODO: Criar um objeto fbeta_score utilizando make_scorer()
scorer = make_scorer(fbeta_score, beta=0.5)

# TODO: Realizar uma busca grid no classificador utilizando o 'scorer' como o método de score no GridSearchCV()
grid_obj = GridSearchCV(clf, parameters, scorering=0)

# TODO: Adequar o objeto da busca grid como os dados para treinamento e encontrar os parâmetros ótimos utilizando fit()
grid_fit = grid_obj.fit(X_train, y_train)

# Recuperar o estimador
best_clf = grid_fit.best_estimator_

# Realizar predições utilizando o modelo não otimizado e modelar
predictions = (clf.fit(X_train, y_train)).predict(X_test)
best_predictions = best_clf.predict(X_test)

# Reportar os scores de antes e de depois
print("Unoptimized model\n------")
print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))

model = DecisionTreeClassifier()

importances = DecisionTreeClassifier().feature_importances_
