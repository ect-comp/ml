# Previsão de situação  de aprovação de aluno com uso de MLP
## Introdução
A atividade foi desenvolvida por Alysson Rafael Oliveira de Lima. Trata-se da necessidade de ter uma previsão o mais cedo possível da probabilidade de reprovação de um aluno, para que seja tomada alguma providência de acompanhamento, para tentar reverter a reprovação. Para isso foi fornecido uma base de dados da turma de Lógica de Programação, da Escola de Ciências e Tecnologia, da UFRN nos períodos de 2017.2 até 2019.1, contendo informações sobre notas de prova, quantidade de questões feitas na prova, quantidades de exercícios submetidos a cada semana dentre outros dados extraídos durante o período.
 
## Metodologia
Para esse experimento foi utilizado o modelo de Machine Learn Multilayer Perceptron - MLP. O perceptron multicamadas é uma rede neural semelhante à perceptron, porém com mais de uma camada de neurônios em alimentação direta, sendo composta por camadas ligadas entre si por sinapses com pesos.
Para o desenvolvimento da rede neural foi utilizado a linguagem de programação Python e o modelo de classificação “**Sequential**” da biblioteca Keras.
Os dados foram divididos em 80% para treinamento e 20% para testes.
Como a proposta é prever a situação de aprovação do aluno com menos tempo possível para que seja realizada alguma intervenção, os atributos foram analisados e escolhidos de acordo com a precedência temporal durante o semestre. Foram realizados vários testes com diferentes quantidades de atributos, chegando a alcançar o melhor resultado, 83% de acertos, utilizando os seguintes atributos:

* **notaProva1** - nota da primeira prova;
* **qsub1** - quantidade de submissões da lista 1;
* **qsub2** - quantidade de submissões da lista 2;
* **qsub3** - quantidade de submissões da lista 3;
* **qsubp1** - quantidade de questões submetidas para a prova 1;
* **igualACeml123** - quantidade de submissões em que o aluno acertou 100%.
	
A rede neural foi construída utilizando 3 níveis. Como camada de entrada um nível com 6 dimensões e 6 unidades de neurônios e função de ativação ‘**relu**’. O segundo nível da rede foi incluído com 15 neurônios e também com a função de ativação ‘**relu**’. A camada de saída foi incluída com uma unidade apenas e com a função de ativação ‘**sigmoid**’.
Para o treinamento da rede, após alguns testes, foi definido o valor 2 para o “**batch_size**” e o valor 40 para “**epochs**”, gerando assim os resultados apontados.
	
## Códigos

Os níveis definidos para a rede MLP.
```py
# Adicionando a camada de entrada e o primeiro nível
classifier.add(Dense( activation = 'relu', input_dim = 6, units = 6, kernel_initializer = 'uniform'))
# Adicionando o segundo nível
classifier.add(Dense( activation = 'relu', units = 15, kernel_initializer = 'uniform' ))
# Adicionando a camada de saída
classifier.add(Dense( activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
```

A configuração do treinamento:
```py
# Treinando a rede
classifier.fit(X_train, y_train, batch_size = 2, epochs = 40)
```

## Conclusão
Através do uso do modelo de rede neural Multicamadas Perceptron, juntamente com os atributos selecionados dos dados fornecidos e das configurações escolhidas para o treinamento da rede foi possível obter uma previsão razoável dos discentes da disciplina de Lógica de Programação. Sendo possível com 83% prever a situação do discente já na primeira unidade e assim haver tempo hábil para poder interferir e analisar formas para que seja possível evitar a reprovação do mesmo. 
