# Identificar e Agrupar Indivíduos de Acordo com Seus Perfis

## Introdução

* O experimento foi realizado por Michel Nunes, discente do curso de Ciência e Tecnologia com Ênfase em Computação Aplicada da UFRN,orientado pelo professor Orivaldo Santana.
* O objetivo é agrupar um indivíduo com determinado perfil a um conjunto de indivíduos com perfis parecidos e que tiveram seus dados previamente analisados.
* A base de dados a ser analisada é composta por diversos registros da quantidade de subimissões semanais dos exercícios propostos pelos professores da disciplina de Lógica de Programação da ECT - UFRN a partir de 2017.2.

## Metodologia 

* O modelo de Machine Learning que está sendo trabalhado é o da Rede SOM (Self Organized Maps, ou Redes Neurais Auto-Organizáveis). É um modelo não supervisionado, ou seja, temos um conjunto de dados e precisamos extrair algo útil desses dados, embora não tenhamos mostrado ao sistema algumas saídas desejadas. É um modelo amplamente utilizado para agrupamento de padrões (Clusters), redução de dimensionalidade, Mineração de dados, Extração de características e Classificação, pois, seu uso como uma ferramenta matemárica é muito interessante para demonstrar dados de baixa dimensionalidade a partir de um conjunto de dados com alta dimensionalidade.
* Foram necessárias algumas instalações de pacotes e alterações na base de dados:
Instalação dos Pacotes da MiniSOM:
```
# Instalando a MiniSOM
!pip install minisom
```
O código a seguir obtem os dados da coluna com valores qualitativos na base de dados e altera seus dados para valores numéricos
```
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dataSet["situacao"] = le.fit_transform(dataSet["situacao"])
```

## Códigos 
Alocando os valores de entrada e saída
```
X = dataSet.iloc[:, 2:23].values
y = dataSet.iloc[:, 25].values
```
Obtendo o tamanho da rede e ativando o treinamento da rede SOM
```
tamanhoXdaRede = 6; 
tamanhoYdaRede = 6; 

quantidadeCaracteristicas = col
from minisom import MiniSom
som = MiniSom(x = tamanhoXdaRede, y = tamanhoYdaRede, input_len = quantidadeCaracteristicas, sigma = 1.0, learning_rate = 0.4)
som.pca_weights_init(X)
```

## Experimentos 

O treinamento com 50000 iterações obteve melhores resultados na classificação dos dados
```
som.train_random(data = X, num_iteration = 50000)
```
* Ao final do treinamento, a rede retorna um vetor de pesos treinados que podem ser utilizados para plotar os gráficos dos dados classificados de forma bidimensional.
```
pesos = som.get_weights()
```
* Os resultados são obtidos através da obtenção e análise dos gráficos. Observe que, no conjunto de gráficos abaixo, aqueles que possuem um "perfil de aprovação", ou algo semelhante a isso, estão mais próximos. O mesmo se observa daqueles com "perfil de reprovação".

![download](https://user-images.githubusercontent.com/55205574/67881284-99d16500-fb16-11e9-8911-416adb6b4004.png)

![download (1)](https://user-images.githubusercontent.com/55205574/67881575-1a906100-fb17-11e9-9cad-58b75ab0d113.png)
