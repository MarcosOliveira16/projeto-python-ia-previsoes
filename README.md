# Projeto Python IA: Inteligência Artificial e Previsões

### Case: Score de Crédito dos Clientes

Você foi contratado por um banco para definir o score de crédito dos clientes. O objetivo é analisar os clientes existentes e, com base na análise, criar um modelo que determine automaticamente o score de crédito de novos clientes como: Ruim, Ok ou Bom.

Arquivos utilizados no projeto: 'clientes.csv' e 'novos_clientes.csv'

---

## Passo a Passo

### **Passo 0 - Entender a empresa e o desafio**

Definição do score de crédito:
- **Good**: Bom
- **Standard**: Ok
- **Poor**: Ruim

---

### **Passo 1 - Importar a base de dados**

Carregamos e visualizamos os dados disponíveis:
```python
import pandas as pd

# Carregando os dados csv
tabela = pd.read_csv("clientes.csv")

# Exibindo os dados
display(tabela)
display(tabela.info())
```

---

### **Passo 2 - Preparar a base de dados para a IA**

Utilizamos `LabelEncoder` para transformar colunas categóricas em valores numéricos:
```python
from sklearn.preprocessing import LabelEncoder

# Codificar colunas categóricas
codificador_profissao = LabelEncoder()
tabela['profissao'] = codificador_profissao.fit_transform(tabela['profissao'])

codificador_credito = LabelEncoder()
tabela['mix_credito'] = codificador_credito.fit_transform(tabela['mix_credito'])

codificador_pagamento = LabelEncoder()
tabela['comportamento_pagamento'] = codificador_pagamento.fit_transform(tabela['comportamento_pagamento'])

display(tabela.info())
```

Definimos as variáveis preditoras (`x`) e a variável alvo (`y`):
```python
# Variáveis preditoras e alvo
y = tabela['score_credito']
x = tabela.drop(columns=["score_credito", "id_cliente"])

# Separar os dados em treino e teste
from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y)
```

---

### **Passo 3 - Treinar a IA**

Criamos dois modelos de machine learning para comparação: `RandomForestClassifier` e `KNeighborsClassifier`:
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

# Criar os modelos
modelo_arvoredecisao = RandomForestClassifier()
modelo_knn = KNeighborsClassifier()

# Treinar os modelos
modelo_arvoredecisao.fit(x_treino, y_treino)
modelo_knn.fit(x_treino, y_treino)
```

---

### **Passo 4 - Escolher o melhor modelo**

Comparamos os modelos utilizando a acurácia:
```python
from sklearn.metrics import accuracy_score

previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)
previsao_knn = modelo_knn.predict(x_teste)

# Acurácia dos modelos
print("Acurácia Árvore de Decisão:", accuracy_score(y_teste, previsao_arvoredecisao))
print("Acurácia KNN:", accuracy_score(y_teste, previsao_knn))
```

Resultados:
- **Árvore de Decisão**: 82,56%
- **KNN**: 74,13%

---

### **Passo 5 - Previsão para novos clientes**

Utilizamos o modelo com melhor desempenho (`RandomForestClassifier`) para prever o score de crédito de novos clientes:
```python
# Carregar novos dados
tabela_novos_clientes = pd.read_csv("novos_clientes.csv")

# Transformar colunas categóricas

tabela_novos_clientes['profissao'] = codificador_profissao.transform(tabela_novos_clientes['profissao'])
tabela_novos_clientes['mix_credito'] = codificador_credito.transform(tabela_novos_clientes['mix_credito'])
tabela_novos_clientes['comportamento_pagamento'] = codificador_pagamento.transform(tabela_novos_clientes['comportamento_pagamento'])

# Previsão
definir_score = modelo_arvoredecisao.predict(tabela_novos_clientes)
print(definir_score)
```

---

## Conclusão

- O modelo de **Árvore de Decisão** foi o mais eficiente, atingindo uma acurácia de 82,56%.
- Ele pode ser utilizado para prever o score de crédito dos clientes com base em dados históricos.

Próximos passos incluem refinar o modelo com mais dados e explorar técnicas de hiperparametrização para melhorar os resultados.

