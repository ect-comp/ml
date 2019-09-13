# USO DE MLP PARA PREVISÃO DE POSSÍVEL REPROVAÇÃO NA DISCIPLINA DE LOP

## Introdução
<p align="justify">O grupo é composto por Jorge Luís e Severino Miguel.
O presente relatório visa resolver o problema de uma possível situação de risco de reprovação de um aluno com o uso de uma base de dados e a ferramenta Multilayer Perceptron, para que possam ser tomadas medidas para prevenir esta reprovação.
Para que seja possível realizar esse processo, utilizamos uma base de dados coletada pelos professores de LOP da Escola de Ciências e Tecnologia da UFRN, de 2017.1 até 2019.1<br></p>

## Metodologia
<p align="justify">Utilizamos a linguagem de programação python para desenvolver os códigos e dentro dessa linguagem a ferramenta escolhida foi o Multilayer Perceptron, que trata-se de uma rede neural com várias camada de neurônios em alimentação direta.
Esse tipo de rede é composta por camadas de neurônios ligadas entre si por sinapses com pesos. Com isso, aplicamos a função StandardScaler, da biblioteca Sklearn.preprocessing, fizemos os testes com 25% dos dados , restando 75% para treino. 
Por fim, para realizar o experimento, selecionamos os dados que foram extraídos no período da primeira unidade
(de 2017.2 até 2019)  e aplicamos o algoritmo com a finalidade de obter uma previsão sobre a situação de algum aluno em um tempo hábil e com isso ser tomado alguma medida à respeito.<br></p>

## Códigos<br>

### Trechos principais:

```py	
#Selecionando quantidade de parâmetros de entrada, primeira camada escondida
classifier.add(Dense( activation = 'relu', input_dim = 6, units = 20, kernel_initializer = 'uniform'))

#Segunda camada escondida
classifier.add(Dense( activation = 'relu', units = 20, kernel_initializer = 'uniform' ))

#Camada de saída
classifier.add(Dense( activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))”
```
```py
#Ajuste do treinamento
classifier.fit(X_train, y_train, batch_size = 5, epochs = 60)
```
  #### Link para código fonte:<br>
<a href='https://colab.research.google.com/drive/1uK15PU4GdH-D1PJSVrfk6O_REiD_G1Wj'>CLIQUE AQUI</a> <br><br>
 
 
 
## Experimentos
 
### Parâmetros manipulados:
* notaProva1 : Nota da prova, unidade 1
* questoesFeitasProva1 : Quantidade de questões feitas na prova, unidade 1 
* qsub1: Quantidade de submissões das listas correspondente (seja lab ou exercício 1)
* qsub2: Quantidade de submissões das listas correspondente (seja lab ou exercício 2)
* qsub3: Quantidade de submissões das listas correspondente (seja lab ou exercício 3)
* igualACeml123: Quantidade de submissões em que o aluno acertou 100%
 
 
<p align="justify">Através da manipulação dos dados, conseguimos prever uma estimativa de 84% das situações finais, podendo assim ajudar de uma forma mais eficiente os alunos na obtenção de um melhor rendimento acadêmico.</p>
