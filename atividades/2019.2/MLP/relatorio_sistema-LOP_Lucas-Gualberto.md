# Multilayer Perceptron do sistema LOP 

## Introdução

O sistema LOP consiste de um site elaborado pela equipe de professores e alguns aluno da Escola de Ciências e Tecnologia (ECT). Essa aplicação tem como objetivo facilitar tanto a vida dos alunos, reunindo várias listas de atividades que são auto corrigidas dando um feedback imediato que aponta onde estão os erros de acordo com vários casos de teste, quanto a vida dos professores, que conseguem ter um tempo de correção de atividades e avaliações bastente reduzido e além disso monitorar a utilização do sistema pelos alunos, podendo fazer uma analise com os dados de turmas, comparar com turmas anteriores entre outras possibilidades. A base de dados usada para esse trabalho foi gerada a partir de dados extraidos desse sistema desde listas de exercícios e laboratório, notas de avaliações, médias, quantidade exercícios feitos por semana, etc. O objetivo geral a ser alcançado é utilizar esses dados para prever a situação de aprovação ou reprovação do aluno da maneira mais rápida possível para que possam ser tomadas medidas ainda durante o semestre para que o aluno possa reverter a situação de reprovação eminente.

## Metodologia 

O modelo de machine learning (ML) usado para a previsão da situação de aprovação ou reprovação dos alunos é o Multilayer Perceptron. Para entender melhor o funcionamento desse modelo é impressindível aprender sobre a ML da qual ele é derivado, o Perceptron. Esse funciona de uma maneira semelhante a um neurônio, onde os dentritos recebem o estimulos elétrico, esses estímulos são processados pelo núcleo do neurônio e depois transmitidos através dos axônios para os dentritos de outra célula nervosa. A sinapse que é a passagem dessa informação de uma célula para outra por meio dos neurotransmissores é regulada pela quantidade desses neurotransmissores presentes em cada terminação dos neurônios. Tal funcionamento é abstraido para o Perceptron em forma de pesos (neurotransmissores) que regulam unidades (neurônios) que podem possuir diferentes funções de ativação. A Multilayer Perceptron difere da Perceptron quanto ao número de camadas existentes. No Perceptron existem apenas 2 camadas, onde uma é de entrada e outra é de saída. Já na MLP além das camadas de entrada e saída existem camadas escondidas que podem tem números variados de unidades. 
A MLP usada para o sistema LOP foi configurada com 3 camadas, sendo a de entrada com duas dimensões e a camada escondida com oito unidades, em que ambas com função de ativação 'relu', e por fim a camada de saída com uma unidade e função de ativação 'sigmoid'.
Para escolha das melhores combinações foram feitos testes de característas de atributos individuais e os dois atributos que foram escolhidos para essa tarefa de previsão foram a nota da primeira unidade (notaProva1) e a quantidade de questões diferentes submetidas nas listas de lab a cada 2 semanas das semanas, mais expecificamente das semanas 4 e 5 (diferentesExerSemanas45). Essa combinação gerou um resultado de 83,33% de acerto de uma base de teste com 90 amostras.

* Explicar o modelo de _machine learning_ (ML) que você está trabalhando. 
* Explicar as etapas do treinamento e teste. 
* Caso tenha selecionado atributos, explicar a motivação para a seleção de tais atributos. 

## Códigos 

```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```
separando 80% da base de dados para treinamento e 20% para testes.\

```python
# Adding the input layer and the first hidden layer
classifier.add(Dense( activation = 'relu', input_dim = 2, units = 8, kernel_initializer = 'uniform'))

# Adding the output layer
classifier.add(Dense( activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
```
criando a camada de entrada com duas dimenções, a camada escondida com 8, ambas com ativação 'relu', e por fim a camada de saida unidimensional e ativação 'sigmoid'.\
```python
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 2, epochs = 50)
```
criando um batch size de 2 e 50 epochs para o processo de treinamento.\


## Experimentos 

Os testes executados foram inicialmente com caracteristicas aleatórias da base de dados a fim de descobrir quais caracteristicas cobinadas geram melhor resposta da rede. Depois de várias mudanças de caracteristicas de entrada as 
que apresentam um amelhor combinação foram notaProva1 e diferentesExerSemanas45, que além de uma boa resposta da rede são parâmetros que podem ser retirados no início do semestre que é um ponto bastante importante para que o professor possa atuar para reverter uma possível situação de reprovação do aluno.\

>Matriz de Confusão:
>[[32  6]
> [ 9 43]]
>Taxa de acerto:
>0.8333333333333334
>90
\
A partir da matriz de confusão podemos concluir a taxa de acerto foi de 83,33% para uma amostra de 90.