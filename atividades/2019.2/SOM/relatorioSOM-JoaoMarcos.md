# Agrupamento e análise de perfil de alunos com rede SOM   

## Introdução
Este trabalho foi desenvolvido por João Marcos Viana Silva, com orientação do prof. Orivaldo Santana. 
A atividade tinha como objetivo analisar a relação existente (ou não) entre a quantidade de questões submetidas por um aluno da disciplina de Lógica de Programação, ministrada na Escola de Ciências e Tecnologia da UFRN, e sua aprovação ou reprovação ao final do semestre. Com isso, seria possível traçar o perfil do aluno e orientá-lo para melhor desempenho na disciplina, além de conseguir uma análise mais efetiva sobre o real impacto que a submissão de questões pelo sistema LOP tem numa eventual aprovação. 
A base de dados utilizada possui 947 entradas, foi gerada e disponibilizada pelo professor e equipe do Sistema LoP, 
nela estão contidas dezenas de atributos relacionados a quantidade de questões submetidas a cada semana pelos alunos que cursaram a disciplina no período entre 2017 e 2019, assim como sua a situação final.

## Metodologia 
O modelo utilizado para o desenvolvimento do algoritmo foi o Self-Organizing Map (SOM), que também pode ser chamado de Mapa de Kohonen, sendo uma rede neural de aprendizagem não-supervisionada. Isto é, a rede não leva em consideração uma saída pré-estabelecida, tendo assim como objetivo analisar as entradas e encontrar padrões entre elas, agrupando-as em grupos com características comuns, o que chamamos de Clusters. As redes do tipo SOM possuem duas camadas em sua estrutura, a primeira delas é a de entrada, responsável por captar os dados iniciais, que podem ter N-dimensões, e aplicando neles atributos matemáticos que chamamos de pesos. A partir disso, o algoritmo irá fazer um mapeamento daqueles dados, formando a segunda camada, ou camada de saída. Esse mapa é basicamente uma matriz, com duas dimensões, onde cada posição representa um grupo de dados similares, ou seja, o algoritmo processa os dados da camada de entrada, procurando padrões, e os posiciona em um índice. A ideia é que os neurônios que estão mais próximos uns dos outros tenham respostas parecidas para entradas semelhantes. Depois, o processo é repetido diversas vezes, enquanto o algoritmo vai ajustando seus pesos e encontrando o índice "vencedor" para cada dado. Ao final, teremos o mapeamento de todos os dados, dispostos nos neurônios de saída. Assim, após esse processo, é possível verificar as características de cada grupo e seus respectivos resultados. Na figura abaixo podemos ter uma representação melhor da estrutura das redes SOM.

!['Representação das camadas da rede SOM'](https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcR1x4NvuAXBPGIhI9a3991XLkEmI3ZGbPuE0CU-6q0oMVOCJXW5)


***Figura 1 - Representação das camadas da rede SOM***

Como a base de dados já era específica, contendo apenas o número de questões submetidas pelos alunos a cada semana, não foi necessário a realização de filtragem nos atributos a serem utilizados no treinamento, ou seja, todas as colunas que continham essas informações foram utilizadas.

## Códigos 
A atividade foi desenvolvida integralmente na linguagem Python, utilizando a biblioteca do Pandas.

* <h3>Normalização do atributo "situacao"</h3>
~~~ python
# Importing the dataset
dataset = pd.read_csv("https://github.com/ect-info/ml/raw/master/dados/lop_submissao_semana.csv");
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dataset["situacao"] = le.fit_transform(dataset["situacao"])
~~~
Logo de início, uma alteração precisou ser realizada no database. O atributo chamado 'situacao' tem seus como seus dados os grupos: 'APROVADO', 'APROVADO POR NOTA', 'REPROVADO POR NOTA', 'REPROVADO' E 'REPROVADO POR NOTA E FALTAS', todas no formato String. Por isso, como podemos ver no código acima, esses dados foram transformados em números inteiros correpondentes. Ou seja, os valores '0', '1', '2', '3', '4', '5', respectivamente substituíram as Strings.

* <h3>Treinamento</h3>
~~~ python
# Training the SOM
tamanhoXdaRede = 6; 
tamanhoYdaRede = 6; 

quantidadeCaracteristicas = col
from minisom import MiniSom
som = MiniSom(x = tamanhoXdaRede, y = tamanhoYdaRede, input_len = quantidadeCaracteristicas, sigma = 1.0, learning_rate = 0.4)
som.pca_weights_init(X)
som.train_random(data = X, num_iteration = 60000)
~~~
A etapa de treinamento da rede SOM foi feita em seguida e alguns testes precisaram ser feitos para decidir os valores para seus parâmetros, eram eles: **tamanhoXdaRede**, **tamanhoYdaRede**, **input_len**, **sigma**, **learning_rate** e **num_iteration**. Após os testes, foi decidido que os melhores resultados foram obtidos com uma matriz quadrada de tamanho 6 como saída, um sigma de 1.0 e taxa de apredizagem de 0.4. O atributo 'input_len' refere-se a quantidade de atributos utilizados, ou seja, a quantidade de colunas que utilizamos na base de dados(21). Além disso, o valor de 60 mil interações foi escolhido para treinar bem a rede, evitando overfitting e underfitting.  

* <h3>Encontrando o neurônio vencedor</h3>
~~~ python
# encontra o vencedor 
x = X[1,:]
pos = som.winner(x)

# matriz de zeros para contador de aprovados 
MContAp = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
# matriz de zeros para o contador de dados em cada neurônio 
MContT = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
cont = 0; 
for x in X: 
  pos = som.winner(x)
  if (Y_train[cont] <2): #Aprovado 
    MContAp[pos] += 1
  MContT[pos] += 1
  cont= cont+1
~~~
Após o processo de treinamento, precisamos analisar a distribuição dos dados entre os neurônios, assim, o código acima nos proporcionou duas matrizes, uma com as posições em 2d dos dados distribuidos nos neurônios, enquanto outra armazenava quantos daqueles dados eram de alunos que foram aprovados.

## Experimentos 
* <h3>Matriz de Neurônios</h3>
~~~ python
Total:
[[ 10.  22.  27.  55.  42.  16.]
 [ 21.  11.  20.  30.  37. 237.]
 [ 10.  21.  16.  19.  20.   5.]
 [ 25.  16.  25.  33.  11.  14.]
 [ 15.  21.   9.  27.  10.  23.]
 [ 25.   8.  16.  17.  15.  19.]]
Aprovados
[[10. 20. 23. 38. 20. 10.]
 [20. 11. 19. 20. 15. 79.]
 [10. 19. 16. 10. 18.  3.]
 [18. 14. 18. 23. 10.  8.]
 [14. 18.  9. 25.  9. 17.]
 [24.  8. 15. 17. 15. 16.]]
~~~
As matrizes acima representam a quantidade de alunos (Total e Aprovados, respectivamente) para os quais o algoritmo relacionou para aqueles neurônios.

!['Matriz'](https://github.com/jota-emi/ML/blob/master/resultado%20neur%C3%B4nios.png?raw=true)

***Figura 2 - Situação dos alunos em cada neurônio***

Acima, podemos observar o resultado do agrupamento da rede, em que cada gráfico representa um neurônio e a proporção entre alunos aprovados e reprovados. A cor azul diz respeito a porcentagem de aprovados, enquanto a laranja, os reprovados.

!['Cada Neurônio viasualizacao'](https://github.com/jota-emi/ML/blob/master/cadaneur%C3%B4nio.png?raw=true)]


***Figura 3 - Quantidade de questões submetidas em média pelos alunos de cada neurônio***

Ainda foi possível criar gráficos para cada neurônio, onde podemos analisar a quantidade de questões submetidas por semana pelos alunos em cada neurônio.

* <h3>Conclusão</h3>

Ao final, podemos observar que a rede conseguiu dividir os alunos com características parecidas. A região direita, sobretudo a superior, da Figura 2 é onde encontra-se o maior número de alunos em situações de reprovação, não por coincidência, verificamos a partir da Figura 3 que tais alunos possuem pouquíssimas submissões. Ainda é possível observar que aqueles neurônios com 100% de aprovações, possuem características comuns de submissões bem regulares, com várias submissões em todas as semanas. Tais analises, deixam claro que existe uma relação de grande influência entre a quantidade e regularidade de questôes submetidas e a eventual aprovação do aluno.

