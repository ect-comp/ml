# Identificador de posições

## Introdução
O Support Vector Machine (SVM) é um conceito utilizado em Machine Learning para identificar padrões em dados e tentar separá-los de acordo com suas classificações. O objetivo dsse trabalho é utilizar um programa de SVM para analisar o posicionamento de uma pessoa por meio de imagens.

### Contribuidores

#### Samuel Amico Fidelis, aluno da Universidade Federal do Rio Grande do Norte do curso de bacharelado em engenharia mecatrônica. Para eventuais dúvidas, entrar em contato pelos
meios abaixo:
- [Github](https://github.com/samuelamico/MachineLearning)
- [Site](https://samuelamico.github.io/)

#### Leonardo Queiroz, aluno de bacharelado em engenharia mecânica na Universidade Federal do Rio Grande do Norte, contato em:
- [Github](https://github.com/leocqueiroz)

#### Gabriel Varela, aluno da Universidade Federal do Rio Grande do Norte do curso de bacharelado em engenharia mecatrônica. Contato em:
- [Github](https://github.com/gabrielvrl)
- [Site](https://gabrielvrl.github.io/)

### Problema
O problema consiste em analisar a imagem de uma pessoa, onde será possível identificar a posição a qual ela está. Há duas posições, sentado e em pé. O programa irá identificar partes predeterminadas do corpo da pessoa e, por meio de um plano bidimensional irá identificar o posicionamento de cada parte. Partes como pé, olho, mão, etc. Identificando corretamente cada posição, faz-se possível estimar o posicionamento da pessoa. O código completo se baseia em uma interface onde o usuário escolhe as posições para serem analisadas. Para cada posição, o usuário deve salvar os pontos para popular o arquivo de treino. Para salvar os pontos, as letras 'c','b' por exemplo, salvam respectivamentes as coordenadas de cada ponto para a classe Cima e Baixo no arquivo de treino. Após salvar a quantidade que o usuário achar boa de pontos ele deve selecionar a opção de validação, onde o modelo de ML SVM é posto em ação e a partir daí toda posição que o usuário fizer o programa irá retornar o nome da posição que ele está fazendo, mediante as  posições que ele mesmo treinou. A figura abaixo ilustra o processo:


![pipline](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/ModeloDiagramaPoseEstimation.png)


### Base de Dados
A base de dados é gerada pelo usuário, quando este seleciona a opção de Salvar Classe. A função de salvar() fica responsável de passar as posições x,y de cada ponto em ordem escolhida pelo usuário e com a devida classe que esse escolheu armazenar. Neste caso a classe é escolhida pelo teclado, onde por exemplo 'b' significa salvar na classe posição baixo.
O arquivo Dados.txt armazena os dados enviados pela função no formato que segue na imagem baixo.

![FuncaoSalvarTXT](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/SalvarTabelaPoseEstimation.png)

## Metodologia 

Modelo de Machine Learning UtilizadoO Machine Learning (ML) utilizado foi o SVM, o qual analisa os dados fornecidos pelo usuário e os separa em duas categorias diferentes, a partir de padrões identificados pelo próprio programa, ou seja, o SVM é um classificador binário não probabilístico. O SVM tende à encontrar uma separação dos dados por meio de uma curva a qual tende à maximizar a
distância entre os pontos mais próximos em relação à cada categoria. De forma resumida, o SVM classifica os dados entre duas configurações distintas e os separa por meio de uma maximização de distância dos pontos para cada categoria.

### Atributos escolhidos
Para realizar a simulação desenvolvida no programa, foram utilizados ao total 12 parâmetros, o qual cada um se refere à um ponto predeterminado para uma parte do corpo da pessoa na imagem. Dentre esses parâmetros, estão:
```
1° - Mão direita
2° - Cotovelo direito
3° - Peito
4° - Cotovelo esquerdo
5° - Mão esquerda
6° - Barriga
7° - Joelho direito
8° - Pé direito
9° - Joelho esquerdo
10° - Pé esquerdo
11° - Coxa direita
12° - Coxa esquerda
```
### Etapas do treinamento
Inicialmente foi dividido os dados fornecidos entre treino e teste e, para isso, foi utilizado um percentual de 75% para treino e os demais 25% para teste.
Após a separação, é utilizado um método de feature scaling para normalizar a distância das variáveis independentes, utilizando a biblioteca sickit-learn. Em seguida é feito um fitting para adequar os dados da fase de treino.

### Etapas do teste
Com o treino realizado, é feito uma predição utilizando a biblioteca do scikit-learn para simular os resultados dos dados de teste. Com o treino realizado, é feita a matriz de confusão a qual será responsável para a visualização do desempenho do algoritmo.

## Experimentos 
Para a realização do teste foram utilizadas 38 imagens, as quais foram analisadas e codificadas para identificar as coordenadas de cada parte do corpo. O rastreamento de cada parte do corpo se dá por meio de um ponto determinado de acordo com os parâmetros mencionados anteriormente.
A execução do algoritmo resulta em duas classes já mencionadas, em pé e sentado. 
Para visualizar o gráfico de resultados do treinamento, foi utilizado o seguinte código:
```py
# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
Xpred = np.array([X1.ravel(), X2.ravel()] + [np.repeat(0, X1.ravel().size) for _ in range(22)]).T
# Xpred now has a grid for x1 and x2 and average value (0) for x3 through x13
pred = classifier.predict(Xpred).reshape(X1.shape)   # is a matrix of 0's and 1's !
plt.contourf(X1, X2, pred,
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))


classe = ['cima','baixo']
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = classe[j])
plt.title('SVM (Training set)')
plt.xlabel('Position')
plt.ylabel('Classe')
plt.legend()
plt.show()
```
Pelo gráfico abaixo mostrado é possível perceber que o código identificou e classificou cada posicionamento de forma coerente, além de os dividir corretamente em dois grupos distintos.

![Grafico](https://github.com/leocqueiroz/MachineLearning/blob/master/SVM/Imagens/Grafico.PNG)

Portanto, com o algorítmo realizado, foi possível separar e classificar cada uma das posições esperadas de acordo com a imagem corretamente, ou seja, o resultado esperado coincidiu com o obtido.
