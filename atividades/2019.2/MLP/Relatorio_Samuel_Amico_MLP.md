# Previsor de situação de risco de alunos

## Introdução
O presente trabalho visa explorar o modelo de Machine Learning para solucionar o problema de prever quais alunos precisam de mais ajuda no semestre
com o objetivo de diminuir a taxa de reprovação.


### Contribuidor
Samuel Amico Fidelis, aluno da Universidade Federal do Rio Grande do Norte do curso de bacharelado em engenharia mecatrônica. Para eventuais dúvidas, entrar em contato pelos
meios abaixo:

* Contato: 
- [Github](https://github.com/samuelamico/MachineLearning)
- [Site](https://samuelamico.github.io/)


### Problema
Uma dos principais objetivos de um professor é ajudar seus alunos, principalmente dar um reforço a mais para aqueles que têm mais dificuldades na máteria. Sendo que muitas vezes
os alunos com dificuldade não chegam a pedir ajuda ao professor, e acabam por desistir ou reprovar na máteria. Portanto o professor não consegue identificar quem são esses alunos, e por isso o algoritmo proposto tenta identificar quem são esses alunos já no começo do semestre para o professor poder ajuda-los.

### Base de Dados
Para solucionar o problema foi proposto analisar dados de turmas de semestres passados da disciplina de Lógica e Programação da Escola de Ciência e Tecnologia da UFRN.
O arquivo de dados consta os seguintes atributos:
```
qsemana == Quantidade de dias diferentes que houve submissão
submeteu == quantidade de questões submetidas pelo aluno por semana
subListaLab == quantidade de submissões na lista de lab a cada 2 semanas
subListaExer == quantidade de submissões na lista de exer a cada 2 semanas
subDistintasLab == quantidade de dias em que submeteu a cada 2 semanas nas listas de lab
subDistintasExer == quantidade de dias em que submeteu a cada 2 semanas nas listas de exer
diferentesLabSemanas == quantidade de questões diferentes submetidas nas listas de lab a cada 2 semanas
diferentesExerSemanas == quantidade de questões diferentes submetidas nas listas de lab a cada 2 semanas
situacao == indica se o aluno foi aprovado (1) ou reprovado (0)
qsub == quantidade de submissões das listas correspondente (seja lab ou exercicio)
qsubp == quantidade de questões submetidas para a prova 1 (L1,L2,L3) ou da prova 2 (L4,L5)
totalsub == total de submissões feitas pelo aluno
igualACem == quantidade de submissões em que o aluno acertou 100%, seja nas L123 ou L45 

```
Onde cada aluno de determinada turma e semestre possui um valor para cada atributo acima. O objetivo é saber escolher também quais serão os atributos mais relevantes
para que se obtenha uma boa resposta logo no ínico do semestre (antes da primeira prova) para que o professor ja auxilie os alunos com dificuldade. Os alunos considerados com
dificuldade são aqueles que reprovaram na disciplina.

## Metodologia 

A primeira etapa foi analisar todos os atributos contidos no arquivo de treinamento. Para cada atributo eu analisei graficamente e por métrica estatisticas a diferença
entre os alunos que foram reprovados e os que foram aprovados. Com base nestas analises foi possível escolher quais atributos tinham maior discrepância entre os reprovados e aprovados. Dois tipos de analise grafica foram feitas conforme mostra as imagens abaixo:


![Histograma](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/Histograma.PNG)

![LinearPlot](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/Plots.PNG)


Para mais informações de como foi feito a seleção dos atributos, olhar o código fonte "AnaliseDadosLOP": [Code](https://github.com/samuelamico/MachineLearning).


Após a análise e escolha dos atributos, foi a hora de utilizar um modelo de machine learning capaz de fornecer uma boa respostar para nosso problema. O modelo escolhido
foi o uma rede neural feed-forward com n entradas (sendo n o número de atributos escolhidos) e uma saída binaria - 0 para reprovado e 1 para aprovado. O modelo da rede neural foi
escolhido devido ao seu bom desempenho em relação aos outros modelos de machine learning, focando assim o problema em saber escolher os melhores atributos para modelar o modelo.

### Atributos escolhidos:
Com base em toda a análise estatística e gráfica os valores de atributos escolhidos foram:
 - 'qsub1','qsub2','qsub3','igualACeml123','submeteu1','submeteu2','submeteu3','subListaExer23','subListaLab23'

### Etapas do treinamento:
A primeira etapa do treinamento é ler a tabela de dados no formato de Dataframe, utilizando para isso a biblioteca pandas do python. Com os atributos escolhidos pode-se
inicializar o vetor de entrada com todas as linhas e apenas as colunas referentes aos atributos. O valor de saída é a situação final do aluno, no caso aprovado ou reprovado.

```py
X = df.loc[:,['qsub1','qsub2','qsub3','igualACeml123','submeteu1','submeteu2','submeteu3','subListaExer23','subListaLab23'] ]
y = df.loc[:,'situacao']
```

Os dados são separados em duas classes: treino e teste. A classe de teste é equivalente a 20% dos dados completos e o restante é treino. Os dados de teste foram escolhidos 
aleatoriamente.

```py
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
```


Para melhorar o desempenho da rede neural é preciso normalizar os dados:

```py
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

Após tratar os dados chegou a hora de colocar na rede neural com 5 camadas escondidas, e escolher apenas as saídas com nível de confiança maior que 0.5:

```py
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor

clf = MLPRegressor(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5,10,10,9,1), random_state=1)

clf.fit(X_train, y_train) 

y_pred = clf.predict(X_test)

y_pred = (y_pred > 0.5)
```

## Experimentos 

Para validar nosso modelo com os atributos escolhidos, uma matriz de confusão foi plotada para analisar os falsos positivos, falsos negativos do dataset de teste.
Com isso foi obtido os seguintes resultados:

```py
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Matriz de Confusão:")
print(cm)
print("Taxa de acerto:")
print((cm[0,0]+cm[1,1])/len(y_test) )


Matriz de Confusão:
[[26 12]
 [24 28]]

 Taxa de acerto:
0.6677
```
Portanto, usando com base as 3 primeiras listas da disciplina obtive 67,7 % de acertos no dado de teste, sem utilizar a nota da primeira prova, pois como só existem duas notas na disciplina inteira, obter a nota inteira da unidade 1 não seria vantagem para uma analise prévia, ja que a primeira unidade é mais da metade do semestre. 