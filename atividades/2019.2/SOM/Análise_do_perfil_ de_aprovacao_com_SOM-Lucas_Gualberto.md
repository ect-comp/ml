# Título do Trabalho 

## Introdução

A base de dados utilizada para esse trabalho é o histórico de submissões de atividades por semana dos alunos na disciplina de LOP.
O objetivo de tratar com essa base de dados é achar o perfil de aprovação e reprovação dos alunos tendo em conta as suas submissões
semanais.  

## Metodologia 

O modelo de machine learning utilizado para a clusterização dos dados de submissões foi a SOM, self-organizing map. Consiste de uma rede neural não supervisionada que é responsável por criar um mapa dimensionado que descretiza a base de dados e a organiza de forma adequada. A SOM conta com um treinamento a partir de valores de entrada escritos na base de dados. Após o processo de treinamento, ele passará por uma fase de mapeamento, a qual será responsável por classificar de forma automática a base de dados em neurônios e organizando esses neurônios. A SOM foi configurada no tamanho de 6 por 6, com uma entrada de tamanho 21, sigma de 1 e taxa de aprendizagem de 0,4, pois apresentaram uma melhor configuração para as saidas. Para o treinamento aleatório foi setado 40000 iterações.

## Códigos 

```python
X_train = dataSet.iloc[:, 2:23].values 
target_train = dataSet.iloc[:,25].values

[row, col] = X_train.shape
```
selecionando os dados para treinamento e o alvo do treinamento.

```python
# Training the SOM
tamanhoXdaRede = 6; 
tamanhoYdaRede = 6; 
quantidadeCaracteristicas = col
from minisom import MiniSom
som = MiniSom(x = tamanhoXdaRede, y = tamanhoYdaRede, input_len = quantidadeCaracteristicas, sigma = 1.0, learning_rate = 0.4)
som.pca_weights_init(X_train)
```
definindo tamanho da rede, tamanho da entrada, sigma e learning_rate

```python
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
  if ((Y_train[cont] == 'APROVADO') or (Y_train[cont] == 'APROVADO POR NOTA')): #Aprovado 
    MContAp[pos] += 1
  MContT[pos] += 1
  cont= cont+1
```
criando matriz de aprovados para cada posição da rede 6 por 6 e matriz de amostras totais por neurônio.

## Experimentos 

* Descrever em detalhes os tipos de testes executados. 
* Descrever os parâmentros avaliados. 
* Explicar os resultados. 
