# Utilização de SVM para classificar a posição de um individuo a partir de imagens 

## Introdução
Este trabalho foi desenvolvido por João Marcos Viana Silva, com orientação do prof. Orivaldo Santana. 
A atividade tinha como objetivo desenvolver um algoritmo capaz de classificar em duas categorias (em pé ou sentado) a posição em que uma pessoa estava em uma foto. Tal classificação abre o caminho para resolução de diversos tipos de problemas, como por exemplo, controle de equipamentos a partir de gestos, uso em videogames ou qualquer outro contexto que análise de posição seja importante. Além disso, a atividade exigiu a criação de uma base de dados própria, processo que é importante na ciência de dados em geral, proporcionando um conhecimento e preparo extra ao aluno. Sendo assim, a base foi gerada a partir de frames de um vídeo produzido pelo autor, aplicadas a uma API do TensorFlow chamada PoseEstimation, capaz de extrair as posições em pixels de onde estariam cada parte do corpo da pessoa na imagem. Ao final, a base continha 128 imagens, em que o individuo estava em pé ou sentado. 

## Metodologia
O modelo utilizado para o desenvolvimento do algoritmo foi o Support Vector Machine (SVM). Este tipo de modelo possui como base a teoria do aprendizado estatístico, ou seja, tem como objetivo encontrar soluções matemáticas para realizar a classificação dos dados. Assim, as SVMs irão mapear todos os dados recebidos, organizando-os em um ambiente multidimensional através de uma função kernel, e a partir disso, calcular a função que melhor divida os dois grupos (linarmente ou não). O processo de escolha dessa função é verificar qual das funções que dividem os dados possui maior distância simétrica entre ela e os dois grupos, o que podemos chamar de margem de divisão. Após o cálculo da margem de divisão, feito e ajustado no processo de treinamento, o procedimento é mais simples, sempre que um novo dado chegar ao algoritmo, o algoritmo irá mapea-lo no ambiente multidimensional e com a margem de divisão vai classifica-lo de X se estiver em um lado da margem, mais próximo dos outros dados classificados como X, ou como Y, se estiver mais próximo de tal grupo.

!['Exemplo gráfico 2D margem de divisão'](https://lamfo-unb.github.io/img/svm/svm1.jpeg)

Como podemos observar na imagem acima, os pontos azuis representam um dos grupos e os amarelos o outro grupo, sendo a linha preta a margem de divisão dos dois, assim, o novo dado a ser mapeado será classificado como azul ou amarelo de acordo com a posição em que irá ser mapeado.

No nosso caso, o processamento dos dados foi feito da seguinte forma, cada imagem possuia atributos correspondentes as posições X e Y de cada parte do corpo utilizada pelo PoseEstimation, funcionando como as dimensões do processo de mapeamento, são elas:

* Nariz;
* Olho esquerdo;
* Olho direito;
* Orelha esquerda;
* Orelha direita;
* Ombro esquerdo;
* Ombro direito;
* Cotovelo esquerdo;
* Cotovelo direito;
* Pulso esquerdo;
* Pulso direito;
* Quadril esquerdo;
* Quadril direito;
* Joelho esquerdo;
* Joelho direito;
* Tornozelo esquerdo;
* Tornozelo direito.

Para o treinamento do algoritmo, isto é, o processo de ajuste da margem de divisão, foram reservadas cerca de 75% das imagens geradas. Ou seja, 3/4 das imagens foram utilizadas para que o algoritmo pudesse compará-las e encontrar padrões entre os atributos de cada uma das classificações, gerando uma função de divisão. Sendo assim, os 25% restantes foram usandos na fase de testes, em que a SVM recebe novos dados e faz a previsão dos mesmos, podendo acertar ou errar.

## Códigos 
A aplicação do SVM foi feita integralmente em Python, utilizando as bibliotecas do Pandas e ScikitLearn. Enquanto o código para gerar a base de dados estava em JavaScript.

<h3>* Separação Treinament/Teste </h3>

~~~ python
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
~~~

25% dos dados reservados para a fase de testes.

<h3>* Escolha dos valores para os atributos gamma e C, processo de treinamento</h3>

~~~ python
# Fitting SVM to the Training set
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
classifier = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf", gamma=0.5, C=50))
))
classifier.fit(X_train, y_train)
~~~ 

Como podemos ver, os valores testados e escolhidos para gamma e C foram: **gamma=0.5** e **C=50** ;

<h3>* Previsão da classificação (em pé ou sentado)</h3>

~~~ python
#Predicting the Test set results
y_pred = classifier.predict(X_test)
~~~

## Experimentos 
<h3>* Matriz de confusão</h3>

~~~ python
Matriz de confusão:
[[18  0] 
 [0 14]]
~~~

A matriz de confusão representa a quantidade de valores para os quais os resultados do algoritmo foram **Verdadeiros Positivos(18)** e **Verdadeiros Negativos(14)** , correspondentes aos acertos e **Falsos Positivos(0)** e **Falsos Negativos(0)** , totalizando uma taxa de acerto de 100%

Com isso, o resultado obtido pela rede neural baseada em SVM com os parâmetros utilizados pode ser considerado excelente. O valor tão expressivo revela que o valores utilizados para gamma e C foram corretos, assim como demonstram uma qualidade enorme da base de dados, que proporcionou resultados limpos. Sendo assim, nas condições em que foram realizadas o experimento, o algoritmo se mostrou extremamente eficaz em classificar se a pessoa na foto está em pé ou sentada.
