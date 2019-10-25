# Estimar Posição de uma Pessoa com Aprendizado de Máquina

## Introdução

* O projeto foi desenvolvido por Michel Nunes, orientado pelo professor Orivaldo Santana, e consiste em desenvolver um algorítmo em python,
utilizando bibliotecas criadas para trabalhar com conceitos de Aprendizado de Máquina, que seja capaz de decidir, através de uma série de
dados previamente apresentados, se uma pessoa está sentada ou em pé. 
* A base de dados foi criada a partir de um vídeo, onde realizo o movimento de sentar-se. A partir disso foram extraídos vários frames
executando o comando mostrado pelo colega Lucas em github.com/Lucasgsr14:
```
$ ffmpeg -i vid.mp4 -r 20 img%d.jpg

```
Onde o vídeo é dividido em 20 frames a cada segundo e a imagem, em formato jpg, é nomeada para img[número], seguindo uma sequência do tipo
img1.jpg, img2.jpg, ... , img145.jpg.
* Mais de 200 frames foram criados, mas, afim de deixar mais preciso, apaguei as imagens as quais considerava como um processo de transição
entre em pé e sentado. Em seguida renomeei as imagens para preencher as lacunas deixadas no processo de remoção. Para recomear as imagens,
basta entrar no diretótio onde elas estão e executar o seguinte código no terminal linux:
```
$ ctd=1 ; for i in *jpg ; do mv $i img${ctd}.jpg ; ((++ctd)) ; done

```
* Tendo as imagens devidamente tratadas em mãos, utilizei a biblioteca Pose Estimation do Tensorflow para estimar a posição (coordenadas
x e y) do nariz, dos olhos, orelhas, ombros, cotovelos, pulsos, quadris, joelhos e tornozelos. As funções da biblioteca foram aplicadas
para cada imagem e seus dados armazenados em formato csv, onde a última coluna é a situação do indivíduo, sendo 1 para em pé e 0 para
sentado.

## Metodologia 

* O Support Vector Machine (SVM), é o modelo de Machine Learning que utilizei nesse projeto. É um algoritmo de aprendizado supervisionado,
no qual se utiliza de uma função kernel para classificar determinados conjuntos de dados, normalmente em dois tipos de características,
mesmo que a partir de uma base de dados multidimensional.
* Logo após a importação das bibliotecas, importação da base de dados e alocamento dos dados para os vetores X e y, precisamos determinar
a quantidade de dados que serão utilizadas para teste e treinamento.
* Inicialmente, foram reservados 75% para treinamento e 25% para testes e o código se mostrou bastante eficaz, tendo uma acurácia de 97%.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

```
* A atribuição de parâmetros para melhor ajustar o SVM ao conjunto de treinamento
```
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

classifier = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=0.25, C=10000))
))

classifier.fit(X_train, y_train)

```
## CÓDIGOS
* Prevendo os resultados do conjunto de testes:
```
y_pred = classifier.predict(X_test)

print(y_test[0:35])
print(y_pred[0:35])
```
```
[1 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 0 1 1 0 0 0 0 1 1 1 1]
[1 1 1 1 0 0 1 1 0 1 1 1 0 1 0 1 1 0 0 1 0 0 0 0 0 1 1 0 1 0 0 1 1 1 1]
```
* A matriz de confusão foi gerada em seguida:
```
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
```
```
[[15  1]
 [ 0 21]]
```

## Experimentos 

* No primeiro momento, executei o programa com os parâmetros descritos acima e o resultado foi a matriz de confusão mostrada anteriormente,
com acurácia de cerca de 97%. O que caracteriza-se como um excelente resultado, provando a eficácia do sistema.
* Para um segundo teste e, sem medo de obter baixa generalização do modelo por ser uma base de dodos relativamente pequena, aumentei a
quantidade de dados para o treinamento para 85%. Deixando assim 15% para os testes. Os resultados foram os seguintes:
Matriz de confusão
```
[[ 9  0]
 [ 0 13]]
```
```
acuraccy = (cm[0][0]+cm[1][1])/(cm[0][0]+cm[0][1]+cm[1][0]+cm[1][1])
print(acuraccy)
```
```
1.0
```
* Significa que, mantendo os parâmetros da função kernel e aumentando um pouco a quantidade de dados para o treinamento, o sistema
consegue um resultado mais preciso, embora o resultado anterior já tenha se mostrado bastante satisfatório.
