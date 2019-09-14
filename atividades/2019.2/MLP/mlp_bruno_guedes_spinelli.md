# Previsão da situação final (aprovação ou não) de alunos utilizando Multi-layer Perceptron (MLP)

## Introdução

Este trabalho foi realizado por Bruno Guedes Spinelli.

O objetivo deste trabalho é realizar uma previsão da situação final, possível aprovação ou reprovação, de alunos a partir de dados coletados sobre as atividades realizadas, ou não, pelo aluno. Para conseguir realizar a previsão foi utilizado Multi-layer Perceptron (MLP), um tipo de rede neural artificial treinada por com uma base de dados feita a partir de alunos reais e durante o semestre de 2017.2. A base de dados utilizada pode ser encontrada clicando [aqui], nela podemos observar as seguintes atributos de dados:

* **notaProva:** nota alcançada pelo aluno na primeira prova nas provas 1 e 2 (notaProva1 e notaProva2).

* **questoesFeitasProva:** questões que o aluno solucionou nas provas 1 e 2 (questoesFeitasProva1 e questoesFeitasProva2).

* **quantidadeProva:** quantidade de questões existentes nas provas 1 e 2 (quantidadeProva1	e quantidadeProva2).

* **qsemana:** quantidade de dias diferentes que houve submissão de questões.

* **submeteu:** quantidade de questões submetidas pelo aluno por semana.

* **subListaLab:** quantidade de submissões na lista de laboratório a cada 2 semanas.

* **subListaExer:** quantidade de submissões na lista de exercícios a cada 2 semanas.

* **subDistintasLab:** quantidade de dias em que submeteu, a cada 2 semanas, nas listas de lab.

* **subDistintasExer:** quantidade de dias em que submeteu, a cada 2 semanas, nas listas de exer.

* **diferentesLabSemanas:** quantidade de questões diferentes submetidas nas listas de laboratório a cada 2 semanas.

* **diferentesExerSemanas:** quantidade de questões diferentes submetidas nas listas de exercícios a cada 2 semanas.

* **situacao:** indica se o aluno foi aprovado (1) ou reprovado (0).

* **qsub:** quantidade de submissões das listas correspondente (seja lab ou exercicio).

* **qsubp:** uantidade de questões submetidas para a prova 1 (L1,L2,L3) ou da prova 2 (L4,L5).

* **totalsub:** total de submissões feitas pelo aluno.

* **igualACem:** quantidade de submissões em que o aluno acertou 100%, seja nas L123 ou L45.

## Metodologia 

O Perceptron foi o primeiro modelo de rede neural artificial criado com a capacidade de aprendizado. Sendo baseado no funcionamento dos neurônios e sinapses biológicas, o Perceptron é capaz de classificar as entradas, de forma binária, em dois grupos distintos e sua arquitetura é baseada em 3 elementos principais:

1) Uma camada de entrada (X).
2) Uma camada de saída (Y).
3) Um conjunto de pesos (W) ligando a entrada à saída.

![perceptron]

**Figura 1 - Representação de um Perceptron**

Neste tipo de rede a aprendizagem da rede é realizada pela modificação dos pesos (W) durante o seu treinamento. Entretanto, o Perceptron apenas é capaz de solucionar problemas linearmente separáveis, o que é um grande problema afinal problemas reais normalmente não podem ser solucionados apenas com uma reta para separar seus elementos. Por este motivo foi escolhido uma rede neural Multi-Layer Perceptron (MLP) para a realização dessa atividade.

O MLP é uma rede neural artificial muito parecida com o Perceptron, porém possui mais de uma camada de neurônios e pode ser utilizada para situações que exijam mais de uma reta para a separação correta dos elementos de entrada. Sua arquitetura consiste nos seguintes elementos:

1) Uma camada de entrada (X).
2) Uma ou mais camadas escondidas (Hidden layers).
3) uma camada de saída (Y).

![mlp]

**Figura 2 - Representação de um MLP**

O principio de funcionamento da MLP é dado pela propagação do sinal de entrada por todas as camadas da rede até que chegue a saída, ou seja, os neurônios das camadas subsequentes utilizam como entrasa o sinal de saída dos neurônios das camadas anteriores, seguida do calculo do erro, utilizando as saídas produzidas pelos neurônios da última camada, e da correção dos pesos de todos os neurônios, a partir da última camada, minimizando seus erros. Desta forma a cada treinamento realizado a rede neural tende a aumentar sua chance de acerto.

Com o objetivo de tentar prever a aprovação ou não dos alunos os seguintes atributos foram escolhidos, com o intuito de indicar se o aluno está, ou não, estudando o conteúdo e realizando as atividasdes passadas com sucesso, ou não:

* **notaProva1**
* **totalsub**
* **igualACeml123**
* **igualACeml45**

## Códigos 

A rede neural foi codificada utilizando a linguagem de programacão Python. O programa utilizou a bibliotéca ```py pandas``` e ```py numpy``` para auxilizar no processo de importar o arquivo de dados e realizar operações matemáticas:

```py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('https://raw.githubusercontent.com/ect-info/ml/master/dados/DataBaseLop.csv')
```
Em seguida foram selecionados quais atributos seriam utilizados para o treinamento e teste da rede neural; sendo eles em X notaProva1 (coluna 2), totalsub (coluna 19), igualACeml123 (coluna 20), igualACeml45 (coluna 21) e em Y situacao (coluna 11); também foi feita a separação dos conjuntos em treinamento e teste utilizando o método ```pysklearn.model_selection``` da biblioteca ```py train_test_split```; onde 20% dos dados foram colocados no conjunto de teste e o restante no conjunto de treinamento; e o ajuste das escalas dos dados, por possuírem diferentes grandezas, utilizando o método ```py sklearn.preprocessing``` da biblioteca ```py StandardScaler```:

```py
X = dataset.iloc[:,[2,19,20,21]].values
y = dataset.iloc[:, 11].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
Logo após foi criada e iniciada a rede neural MLP utilizando a biblioteca ```py keras```, em seguida foram adicionadas a camada de entrada, as duas camadas escondidas e a camada de saída, além da seleção dos modos de operação da rede:

```py
import keras
from keras.models import Sequential
from keras.layers import Dense

Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense( activation = 'relu', input_dim = 4, units = 4, kernel_initializer = 'uniform'))

# Adding the second hidden layer
classifier.add(Dense( activation = 'relu', units = 6, kernel_initializer = 'uniform' ))

# Adding the output layer
classifier.add(Dense( activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
```

Por último, a rede neural foi treinada utilizando o conjunto de dados de treino, a realização do teste da rede neural e sua validaçao por meio do cálculo de precisão e sua matriz de confusão:

```py
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 5, epochs = 30)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(y_pred[0:10])

y_pred = (y_pred > 0.5)
print(y_pred[0:10])

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
``` 

## Experimentos 

Após ser treinada a MLP foi testada utilizando um conjunto de teste definido aleatoriamente. Para realizar esse teste os dados de entrada (X do conjunto de teste) do conjunto de teste foi inserido na rede neural gerando saidas de predição (Y da predição) e então se comparou os dados de saída do conjunto de teste (Y do conjunto de teste e situação final do grupo de alunos inseridos no conjunto de teste). Assim foi gerada uma matriz de confusão e consequentemente um percetual de acerto da MLP desenvolvida.

**Matriz de Confusão:**

[31 7]

[9 43]

**Taxa de acerto:**
0.8222222222222222 ou 82.2%

<!-- Links -->

[aqui]: https://github.com/ect-info/ml/blob/master/dados/DataBaseLop.csv
[perceptron]: https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Sadssa.png/469px-Sadssa.png
[mlp]: https://1.bp.blogspot.com/-Xal8aZ5MDL8/WlJm8dh1J9I/AAAAAAAAAo4/uCj6tt4T3T0HHUY4uexNuq2BXTUwcChqACLcBGAs/s1600/Multilayer-Perceptron.jpg
