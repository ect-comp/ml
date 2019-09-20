# Previsão de Aprovação do Discente em LOP com MLP

## Introdução

  O documento presente foi desenvolvido pelo bacharelando em Ciência e Tecnologia com ênfase em Computação Aplicada da Universidade Federal do Rio Grande do Norte (UFRN), Michel Nunes.
  Tráta-se da necessidade de uma previsão de situação de risco de um aluno na disciplina de Lógica de Programação da Escola de Ciência e Tecnologia baseada no seu desempenho nas primeiras semanas no curso.
  A base de dados foi disponibilizada pelo professor Orivaldo Vieira e é composta com diversos campos que registram dados como: Notas das provas da primeira e segunda unidades; Quantidade de questões em cada prova; Quantidade de questões feitas por cada discente; Quantidade de questões submetidas em cada lista de exercícios; Dentre outros atributos relevantes à análise.

## Metodologia 

  O modelo de  _machine learning_ utilizado foi o MultLayer Perceptron (MLP), que é o método de criar uma rede neural artificial no qual utiliza-se de um estímulo que iniciará uma atualização de parâmetros até atingir o critério de parada. Cada atualização passa pela regra de correção de erros, procurando diminuí-los a cada iteração. Sendo assim, de fato, o modelo aprende com os testes escolhidos.
  Foi utilizada a classe train_test_split da biblioteca sklearn.model_selection para realizar o treinamento e teste, onde 20% (vinte por cento) dos dados foram escolhidos para o teste.
  Os atributos foram selecionados com o palpite de que a aprovação do aluno na disciplina depende do seu aprendizado e seu aprendizado depende da sua prática, já que se trata de uma disciplina onde a prática é o que leva ao sucesso. Considerando a quantidade de exercícios propostos, a primeira unidade possui informações suficientes para uma análise de risco.

## Códigos 
```py
# CONFIGURANDO A CAMADA DE ENTRADA COM TRÊS NEURÔNIOS, MAIS TRÊS NA CAMADA ESCONDIDA
classifier.add(Dense( activation = 'relu', input_dim = 3, units = 3, kernel_initializer = 'uniform'))


### UMA CAMADA ESCONDIDA COM SEIS NEURÔNIOS
classifier.add(Dense( activation = 'relu', units = 6, kernel_initializer = 'uniform' ))


### CAMADA DE SAÍDA - APENAS UMA SAIDA
classifier.add(Dense( activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))

### Ajuste do RNA ao conjunto de treinamento
classifier.fit(X_train, y_train, batch_size = 5, epochs = 40)
```
## Experimentos 

  Os parâmetros avaliados foram as notas da primeira prova, a quantidade de submissões de questões da lista até a primeira prova e a quantidade de submissões até a primeira prova as quais o aluno acertou 100%.
```py
# Parâmetros de entrada e saída
X = dataset.iloc[:,[2,17,20]].values
y = dataset.iloc[:, 11].values

### Saída
Taxa de acerto:
0.8333333333333334
```
## Conclusão
  De acordo com a saída apresentada, o modelo de machine learning utilizado tem 83% de chance de prever a aprovação de um aluno na disciplina de LOP avaliando seu desempenho na primeira unidade (parâmetros).
