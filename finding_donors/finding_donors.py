# Importe as bibliotecas necessárias para o projeto.
import numpy as np
import pandas as pd
from time import time
#from IPython.display import display # Permite a utilização da função display() para DataFrames.

# Importação da biblioteca de visualização visuals.py
#import visuals as vs

# Exibição amigável para notebooks
# %matplotlib inline

# Carregando os dados do Censo
data = pd.read_csv("census.csv")

# Sucesso - Exibindo o primeiro registro
print(data.head(n=1))

# TODO: Número total de registros.
n_records = data.shape[0]

# TODO: Número de registros com remuneração anual superior à $50,000
n_greater_50k = data[data.income == '>50K'].shape[0]

# TODO: O número de registros com remuneração anual até $50,000
n_at_most_50k = data[data.income == '<=50K'].shape[0]

# TODO: O percentual de indivíduos com remuneração anual superior à $50,000
greater_percent = (n_greater_50k/n_records)*100

income_raw = data['income']

# Exibindo os resultados
print("Total number of records: {}".format(n_records))
print("Individuals making more than $50,000: {}".format(n_greater_50k))
print("Individuals making at most $50,000: {}".format(n_at_most_50k))
print("Percentage of individuals making more than $50,000: {:.2f}%".format(greater_percent))

income = income_raw.replace('>50K', 1).replace('<=50K', 0)
print(income)