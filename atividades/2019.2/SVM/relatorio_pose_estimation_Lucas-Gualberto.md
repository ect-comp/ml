# Classificação de Movimentos com o PoseNet usando Support Vector Machine (SVM)

## Introdução

O PoseNet é uma modelo de visão que pode ser usado para estimar a pose de uma pessoa em uma imagem ou video estimando onde estão as aticulações chaves do corpo estão. Em um primeiro modelo de captação de dados foi usado um código em html com javascript que retornava como principal dado um vetor de posições para cada imagem na pagina do browser usado para executar o código. No entanto, esse método nem sempre funcionava para configurações de computadores diferentes. Então a solução foi fazer um codigo em node.js que retorna um array de posições quando recebe um URL de um imagem. Com isso foi possível criar um código em python que faz requisições http, submete um vetor de URL's e recebe os vetores de posição para cada imagem e por fim cria um arquivo de extensão '.csv' para a base de dados. O objetivo principal dessa base de dados é treinar uma SVM para identificar, entre cinco posições (levantado, pé direito levantado, pé esquerdo levantado, mão direita levantada e mão esquerda levantada), qual a imagem enviada corresponde. A base de dados utilizada nessa SVM possui 977 imagens e cada imagem gera 17 coordenadas do local das juntas.

## Metodologia 

A SVM é um modelo de machine learning que consegue aplicar uma tranformação no espaço de dados dessa forma conseguindo separar dados que aparentemente não aparentam ser não lineramente separáveis, mas que em algum plano do espaço se tornam linearmente separáveis. Ela utiliza funções de Kernel, em SVMs não lineares, para calcular hiperplanos ótimos que consigam separar da melhor forma os dados fornecidos no processo de treinamento. Com essa ML foram usados 20% da base de dados para treinamento e 80% para teste. Outros pontos importantes de ajuste para esse modelo de ML são os parâmetros gamma, que define o quão ajustado o modelo deve ser, e o c, que define a quantidade de curvas a SVM pode realizar. Para essa experiência foram utilizados um gamma de 0.009 e um c de 4.

## Códigos 

```python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8, random_state = 0)
```
separando 20% da base de dados para treinamento e 80% para testes.\

```python
# Fitting SVM to the Training set
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

#classifier = SVC(kernel = 'linear', random_state = 0)
classifier = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=0.009, C=4))
))

```
fazendo a SVM definindo o kernel como 'rbf' o gamma como 0,009 e o C como 4\

## Experimentos 

Com as configurações feitas na SVM o modelo gerou uma acurácia de 99,23% para o vetor de teste com 781 amostras\

>[[175   0   0   0   0]
> [  0 209   0   0   0]
> [  3   0  80   3   0]
> [  0   0   0 175   0]
> [  0   0   0   0 136]]
>
> 0.9923175416133163
