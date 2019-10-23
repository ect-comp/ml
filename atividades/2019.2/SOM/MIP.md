# Mapeamento de Índice de Aprovação
## Introdução
Este trabalho consiste em analisar um banco de dados pelo método SOM de machine learning.
O problema em questão consiste em analisar as primeiras semanas de um semestre em uma matéria e tentar determinar as situações de cada discente, dividindo eles em perfis de caracterização.
Cada perfil possui uma parcecla de aprovado e de reprovado. O código será responsável por dividir cada usuário dentro dessas parcelas, podendo assim entender melhor o comportamente de cada aluno ao longo da disciplina.
A base de dados consiste em descrever as notas de cada atividade semanal submetida de cada aluno. Contendo um total de 21 semanas e, consequentemente, 21 atividades.
Foi utilizado, também, uma árvore de decisões por fins de comparação, para analisar o comportamento lógico comparado ao código SOM realizado.

### Contribuidores

- Samuel Amico Fidelis, aluno da Universidade Federal do Rio Grande do Norte do curso de bacharelado em engenharia mecatrônica. Para eventuais dúvidas, entrar em contato pelos
meios abaixo:
* [Github](https://github.com/samuelamico/MachineLearning)
* [Site](https://samuelamico.github.io/)

- Leonardo Queiroz, aluno de bacharelado em engenharia mecânica na Universidade Federal do Rio Grande do Norte, contato em:
* [Github](https://github.com/leocqueiroz)

- Gabriel Varela, aluno da Universidade Federal do Rio Grande do Norte do curso de bacharelado em engenharia mecatrônica. Contato em:
* [Github](https://github.com/gabrielvrl)
* [Site](https://gabrielvrl.github.io/)

## Metodologia

### SOM
O SOM, self-organizing map, é uma rede neural artificial sem supervisionamento que é responsável por criar um mapa dimensionado que descretiza a base de dados, além de organizá-las de forma adequada
A base SOM possui uma fase de treino, a qual será responsável por treinar a rede neural a partir dos valores de input escritos na base de dados. Após o processo de treino, ele passará por uma fase de mapeamento, a qual sera responsável por classificar de forma automática a base de dados em neurônios, organizando esses neurônios de forma automática.

#### Experimento e Códigos

O código para a resolução desse problema pode ser encontrado no Github de Gabriel: [Github](https://github.com/gabrielvrl/Machine-Learning-ECT/blob/master/SOM_U2.ipynb).

Primeiro passo é importar as bibliotecas a serem utilizadas e o banco de dados, vamos utilizar o pandas pois é uma biblioteca que fornece ferramentas de análise de dados e estruturas de alta performance e fáceis de usar. E também o numpy, para permitir trabalhar com arrays, vetores e matrizes.
Assim, utilizamos o "read_csv" da biblioteca pandas para fazer a leitura dos dados do repositório do Github.

```py
import pandas as pd
import numpy as np
dataSet =  pd.read_csv("https://raw.githubusercontent.com/ect-info/ml/master/dados/lop_submissao_semana.csv",index_col=False )

dataSet['situacao'] = pd.Categorical(dataSet['situacao'])
dataSet['situacao'] = dataSet['situacao'].cat.codes

dataSet.head()
```

Seguido da instalação da MiniSOM, a MiniSOM é uma implementação minimalista dos Self-Organizing Map(SOM)

```py
!pip install minisom
```

Dessa forma, podemos realizar o treinamento da nossa rede:
Treinamos com todas as variáveis semanais, como "semana 1, semana 2, semana 3,..., semana 21".

```py
X_train = dataSet.iloc[:,2:23].values 
target_train = dataSet.iloc[:,25].values

[row, col] = X_train.shape
print (row," ",col)

print(X_train[1,:])
```

Definimos o tamanho em X e Y da rede (7x7) e utilizando o minisom, passamos os parâmetros (x,y,input_len, sigma e learning_rate), assim, fazemos o treinamento.

```py

# Training the SOM
tamanhoXdaRede = 7; 
tamanhoYdaRede = 7; 

quantidadeCaracteristicas = col
from minisom import MiniSom
som = MiniSom(x = tamanhoXdaRede, y = tamanhoYdaRede, input_len = quantidadeCaracteristicas, sigma = 1.0, learning_rate = 0.4)
som.pca_weights_init(X_train)

som.train_random(data = X_train, num_iteration = 40000)
```

Como no método SOM não existe uma fase, de fato, de teste, apenas uma análise de dados, utilizamos o código abaixo que será responsável por criar nossa matriz resultado. Essa matriz é o mapeamento posto em prática. Ela será responsavel por identificar os perfis de cada usuário e dividí-los entre as categorias.

```py
# encontra o vencedor 
x = X_train[1,:]
pos = som.winner(x)


# matriz de zeros para contador de aprovados 
MContAp = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
# matriz de zeros para o contador de reprovados 
MContT = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
cont = 0; 
for x in X_train: 
  pos = som.winner(x)
  if (Y_train[cont] <= 1): #Aprovado 
    MContAp[pos] += 1
  MContT[pos] += 1
  cont= cont+1
```

Após a criação da matriz é separado as porcentagens entre Aprovado e Reprovado para cada perfil, chegando no resultado disponível na imagem abaixo:

![AprovadoReprovado](https://github.com/leocqueiroz/MachineLearning/blob/master/SOM/Imagens/AprovadoReproado.PNG)

Onde, na imagem, a cor azul representa Aprovação e a laranja Reprovação.

Após essa divisão, é possível encontrar o mapeamento de pesos, a partir do código abaixo:

```py
# Mostra todos os pesos 
cont = 1;
x = np.arange(quantidadeCaracteristicas)
for row in pesos:
  for elem in row:
    plt.subplot(tamanhoXdaRede,tamanhoYdaRede,cont)
    cont=cont+1
    plt.axis([-1, 6, 0, 10])
    plt.bar(x, elem)
    plt.plot([-1,6],[5,5],'r')
plt.show()
#plt.savefig("test_som.jpg", dpi=150)
```

O resultado dos pesos desse mapeamento é mostrado na seguinte imagem

![Pesos](https://github.com/leocqueiroz/MachineLearning/blob/master/SOM/Imagens/Pesos.PNG)

Sendo assim, então, possível observar os resultados do SOM, uma vez que o mesmo não possui de fato uma fase teste, e sim uma fase de análise do mapeamento gerado.

### DecisionTree

Árvores de decisão são métodos de aprendizado de máquinas supervisionado não-paramétricos, muito utilizados em tarefas de classificação e regressão. Árvores, de um modo geral em computação, são estruturas de dados formadas por um conjunto de elementos que armazenam informações chamadas nós.
Em uma árvore de decisão, uma decisão é tomada através do caminhamento a partir do nó raiz, maior nível hierárquico (o ponto de partida) até o nó folha ou terminal.
A árvore de decisão foi utilizada para comparar os resultados obtidos utilizando o modelo SOM anteriormente.

#### PowerBI

Foi utilizado o PowerBI para construir um painél para que se pudesser analisar e escolher os melhores dados de semanas para treinar o modelo de árvore de decisão. A métrica utilizada levou em consideração as primeiras semanas 
, pois é preciso descobrir os alunos que estão em dificuldade logo para que se possa dar uma atenção melhor neles, logo em seguida foi analisado no painel PowerBI que as 5 ou 6 primeiras semanas demonstraram
um comportamento bem diferente entre elas.

O painel do PowerBI está disponivel pelo link: [PowerBI](https://github.com/samuelamico/MachineLearning/tree/master/DecisionTree)

#### Experimento e Códigos

Escolhido as semanas para serem treinadas através de funções da biblioteca sklearn:

```py
from sklearn.model_selection import train_test_split # Import train_test_split function

x = dataset_filter.loc[:,[dataset_filter['semana 1'],dataset_filter['semana 2'],dataset_filter['semana 3'],dataset_filter['semana 4'],dataset_filter['semana 5'],dataset_filter['semana 6']]]
y = dataset_filter.iloc[:,-1]
feature_cols = ['semana 1','semana 2','semana 3','semana 4','semana 5','semana 6']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1) 
```

Então logo em seguida foi criado o objeto árvore de decisão e utilizando o dataset filtrada para as semanas mais importantes temos:

```py
LopTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
LopTree # it shows the default parameters
LopTree.fit(X_train,y_train)
```

Foi obtida então a seguinte acurácia para o uso de árvore de decisão:

```py
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, predTree))

Accuracy: 0.4456140350877193
```

Portanto, a acurácia não foi tão satisfatória, porém foi testado utilizando outras combinações de semanas e percebeu que 
realmente as seis primeiras semanas eram as melhores possiveis escolhas tomando como base o resultado de acurácia, como ja indicava no painél do PowerBI. Portanto mesmo que a acurácia não tenha sido tão satisfatoria utilizando o método de árvore de decisão
vale ressaltar a importância deste resultado de acurácia, pois mostra que realmente as seis primeiras semanas são as mais importantes para se obter um melhor resultado de previsão.

A imagem abaixo mostra a árvore de decisão:

![Arvore](https://github.com/samuelamico/MachineLearning/blob/master/DecisionTree/loptree.png)