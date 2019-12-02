# USO DE MLP PARA IDENTIFICAR PREDISPOSIÇÃO À DIABETES

## Introdução
<p align="justify">O grupo é composto por Jorge Luís e Severino Miguel.
<br>
O presente relatório visa identificar uma possível situação de risco de uma pessoa que possa conter sinais de que tem diabetes. 
Com o uso de uma base de dados e a ferramenta Multilayer Perceptron, para que possam ser tomadas medidas preventivas.
Utilizamos uma base de dados dataset Pima Indians, trata-se de um estudo feito com índios da tribo PIMA dos Estados Unidos.<br>
</p>
 
## Sobre os índios Pima
 <p align="justify">No final do século 19, a água que os índios Pima usavam para irrigação foi desviada por fazendeiros, 
 resultando em uma carência de alimento e desnutrição para a tribo. 
 Na tentativa de contornar a situação, o governo americano começou a fornecer comida a eles: 
 toucinho, farinha, açúcar, entre outros suprimentos. Isso fez com que eles se tornassem sedentários. 
 A porcentagem de gordura na dieta, que originalmente era de 15%, passou a 40%. 
 Resultado: os Pima se tornaram o grupo mais obeso dos Estados Unidos e o que mais tem diabetes. 
 Metade dos Pima são diabéticos.</p>
 <br>
 
 ![Tabela](https://www.musculacao.net/wp-content/uploads/2012/01/indios_pima.jpg)
 <br>
 Índios Pima. Fonte: <https://www.musculacao.net/wp-content/uploads/2012/01/indios_pima.jpg>

## Metodologia
<p align="justify">Utilizamos a linguagem de programação python para desenvolver os códigos e dentro dessa linguagem a ferramenta escolhida foi o Multilayer Perceptron, que trata-se de uma rede neural com várias camada de neurônios em alimentação direta.
Esse tipo de rede é composta por camadas de neurônios ligadas entre si por sinapses com pesos. Com isso, aplicamos a função StandardScaler, da biblioteca Sklearn.preprocessing, fizemos os testes com 25% dos dados , restando 75% para treino. 

## Códigos<br>
 
### Primeira parte
 
```py	
# Rede Neural

# Parte 1 - Processamento de Dados

# Importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```
```py
#Ajuste do treinamento
# Importando a base de dados
dataset = pd.read_csv('https://raw.githubusercontent.com/jorgelgf/ML/master/projetoFinal/base/diabetes.csv')


dataset.head( 10 )
```
<br>
A base de dados consiste em mulheres acima dos 21 anos, ao todo foram 769 para a base de dados original. Os parâmetros são:
<br><br>
- Número de vezes que esteve grávida;
<br>
- Concentração de glicose no plasma a 2 horas em um teste oral de tolerância à glicose;
<br>
- Pressão ArterialDiastólica (mm Hg);
<br>
- Espessura da dobra da pele do tríceps (mm);
<br>
- Insulina sérica de 2 horas (mu U / ml);
<br>
- Índice de massa corporal;
<br>
- Pré-disposição genética para ter diabetes;
<br>
- Idade;
<br>
- Variável de saída, resultado, 0 para não tem diabetes e 1 para tem diabetes;
<br><br>
Foram feitas manipulações nos dados originais, alguns membros não continham os valores referentes à insulina. Para obter uma fidelidade maior
com o uso do algoritmo, foi retirado os dados que não continham informações sobre os nívies de insulina, totalizando com 395 pessoas na base de dados nova.

<br><br>

![Tabela](https://github.com/jorgelgf/ML/blob/master/projetoFinal/imagens/basedados.png?raw=true)
Imagem da base de dados. Fonte: <https://colab.research.google.com/drive/10jrqzfNk5dQFbO0ZGAVYCOIFUon1dqiu#scrollTo=3ty937PWG8mp>
<br>
<br>
Selecionando os parametros:
<br>
```py
X = dataset.iloc[:,0:8].values
y = dataset.iloc[:, 8].values
print(X.shape)
```
<br>


```py
# Dividindo a base de dados para teste e treino 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```
<br>

```py
# Escalando recursos
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

<br>

### Segunda parte

<br>

```py
# Parte 2 - Fazendo a Rede Neural

# Importando a bilbioteca Keras e os packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializando a rede Neural
classifier = Sequential()

```
<br>

```py
# Input layer e primeira hidden layer
classifier.add(Dense( activation = 'relu', input_dim = 8, units = 15, kernel_initializer = 'uniform'))

# Segunda hidden layer
classifier.add(Dense( activation = 'relu', units = 15, kernel_initializer = 'uniform' ))


# Output layer
classifier.add(Dense( activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))



# Compilando a Rede neural
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Ajustando a Rede neural ao conjunto de treinamento
classifier.fit(X_train, y_train, batch_size = 3, epochs = 90)

```
<br>


### Terceira parte
<br>

```py
# Parte 3 - Fazendo as predições e avaliando o modelo

# Prevendo os resultados do conjunto de testes
y_pred = classifier.predict(X_test)
print(y_pred[0:10])

y_pred = (y_pred > 0.5)
print(y_pred[0:10])

# Fazendo a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print("Taxa de acerto:")
print((cm[0,0]+cm[1,1])/len(y_test) )

Taxa de acerto:
0.8282828282828283
```

<br>


## Conclusão
<br>

<p align="justify">Obtivemos uma boa taxa de acerto de aproximadamente 83%. Podemos concluir que para ter uma maior fidelidade no resultado seria necessário
uma base de dados maior e mais consistente, com dados mais atualizados. Mesmo assim, tendo em vista o uso dessa ferramente, é possível obter uma possível previsão
para que possa ser tomadas medidas preventivas podendo gerar uma melhor qualidade de vida para uma pessoa.


</p>
<br>  

 #### Link para código fonte 

<a href='https://colab.research.google.com/drive/10jrqzfNk5dQFbO0ZGAVYCOIFUon1dqiu#scrollTo=Y2W85-LdSsiU'>CLIQUE AQUI</a> <br><br>

  #### Link para referências
<br>

Acessado em 2019: <https://www.kaggle.com/uciml/pima-indians-diabetes-database>

<br>

Acessado em 2019: <http://www.academia.org.br/artigos/diabetes-aprendendo-com-os-indios>


 

 
