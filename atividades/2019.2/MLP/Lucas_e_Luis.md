# Interpretando dados educacionais usando Multi-layer Perceptron(MLP) 

## Introdução
<p align = "justify">
No desenvolvimento do relatório houve a colaboração direta de dois discentes, sendo eles Luis Felipe Vanin Martins e Lucas Figueredo Varela Alves.
<p align = "justify">
Em diversas áreas da ciência existem problemas extremamente complexos que requerem uma quantidade muito grande de dados que geram muitos problemas para a manipulação humana, por isso são utilizados meios computacionais para resolver tais problemas. Além da grande quantidade de dados, existem problemas reais que não admitem resoluções simples e, assim, para soluciona-lo não se pode usar algoritmos convencionais, uma das maneiras para resolver esses celeumas é utilizar de métodos mais avançados que, em questão, serão algoritmos de aprendizado de máquina(machine learning).
<p align = "justify">
No relatório em questão, veremos um problema relacionado a previsão resultado final de alunos a partir de dados coletados acerca de suas atividades realizadas. Devido a grande quantidade de dados e pela dificuldade em prever padrões humanos, é impossível aplicar um algoritmo simples para prever o resultado de cada aluno, logo, serão utilizados métodos de machine learning(ML).
<p align = "justify">
O banco de dados, como supracitado, apresenta diversas atividades feitas por um aluno e seus respectivos desempenhos na matéria de Lógica de programação(LOP) em um determinado semestre, tais dados foram levantados com a ajuda da plataforma LOP e fornecida pelo professor Orivaldo Vieira de Santana.
</p>

## Metodologia 

<p align = "justify">
Para tentar prever o resultado final de um aluno, caso ele seja reprovado ou aprovado, será utilizado o método de ML denominado MultiLayer-perceptron(MLP) que é um sistema de múltiplas camadas de perceptron. Para explicar esse método é logicamente necessário explicar o método de perceptron.
<p align = "justify">
O perceptron foi o primeiro método de rede neural com a possibilidade com a capacidade de aprender, logo, fica claro o por quê a escolha deste método, pois podemos separar parte do banco de dados para aprender que será denominado a parte de treino dos dados selecionados. O perceptron busca simular um neurônio em seu funcionamento, quando criamos diversas camadas de perceptron criamos uma situação análoga ao funcionamento cerebral(de maneira muito simplória quando se comparada com o cérebro).
<p align = "justify">
O funcionamento básico do perceptron consiste na existência de uma camada de entradas(inputs) que são os parâmetros para análise dos dados usados pelo algoritmo, uma camada que contém um sistema de pesos a qual cada input é submetido fazendo com que os dados sejam manipulados de maneira a encontrar uma “resposta” e mais uma camada de saída(outputs) que representa a predição do perceptron em relação aqueles parâmetros inseridos na primeira camada(inputs). Então, no banco de dados em questão, as atividades feitas pelos alunos e seus resultados serão passados como inputs na primeira camada e logo em seguida será para a camada intermediária(que é nomeada como hidden layer), onde esses dados se relacionam com os pesos, e depois do processo que ocorre no hidden layer a última camada tem a função binária de prever se o discente irá reprovar ou passar.
</p>
<center>
    <img src = "https://akashsethi24.files.wordpress.com/2017/09/perceptron.png?resize=385%2C254">
    </img>

<h5>(fig 1 - esquematização do perceptron e suas camadas)</h5>
</center>

<p align = "justify">
Tal processamento que ocorre no hidden-layer que busca prever os resultado do aluno é desenvolvido pelo próprio algoritmo por um sistema de aprendizagem de RNA , a qual ocorre no procedimento de treino. No período de treinamento se utiliza os pesos pré-definidos para se abstrair as previsões dos resultados finais dos alunos, sendo que essa previsão será sempre comparada com o resultado real, ou seja, a parte do banco de dados usada para o  treinamento deve ter seus resultados finais já disponíveis, e após a comparação é gerado um erro. Esse erro será utilizado para modificar os valores dos pesos durante a fase de treino e, assim, buscando otimizar os valores dos pesos para melhor prever o cenário. Após o treinamento, é feito a parte de teste a qual outra parcela do banco de dados é selecionada para ser usado, porém não há modificação nos pesos, logo, tenta prever novos dados a partir do que ele teria trabalhado com outros dados similares.
<p align = "justify">
Para a construção de um MLP consiste um perceptron com mais de um hidden-layer.
</p>
<center>
    <img src ="https://www.researchgate.net/profile/Mohamed_Zahran6/publication/303875065/figure/fig4/AS:371118507610123@1465492955561/A-hypothetical-example-of-Multilayer-Perceptron-Network.png" width = "400" height = "">
</img>
</center>
<center>
<h5>(fig 2 - representação do MLP, sendo cada seta contendo seu respectivo peso)</h5>
</center>

<p align = "justify">
Retornando ao nosso problema relacionado a previsão no resultado final, temos a intenção de utilizar de conseguir visualizar a futura situação do aluno com o mínimo de atributos e de maneira mais rápida. É necessário utilizar o mínimo de parâmetros possíveis, pois caso se use muito parâmetros a fase de treino pode entrar em um caso de overfit . Quando se trata de conseguir os resultados mais rápido se dialoga com o fato de prever a futura situação do aluno com a finalidade de tentar reverter essa possível situação do aluno quando se é tem o quadro negativo, logo, vamos buscar aproveitar dos atributos que são adquiridos até a primeira prova(prova1), pois caso contrário a previsão pode vir de maneira tardia, ou seja, gerando um quadro irreversível.   
</p>

## Códigos 

<p align = "justify">
O código gerado é elaborado na linguagem de programação Python, aproveitando de suas bibliotecas que dão suporte ao machine learning. As bibliotecas empregadas serão o numpy, pandas, keras e sklearn, sendo essas últimas duas dando suporte diretamente ao ML.
</p>
~~~ python
url = "https://raw.githubusercontent.com/ect-info/ml/master/dados/DataBaseLop.csv"
dataframe = pd.read_csv(url)
~~~
<p align = "justify">
As próximas linhas de código importantes são relacionados ao recorte do banco de dados em uma variável. Os dados atribuídos à variável “x” são os parâmetros utilizado para prever o resultado final e já na variável “y” são os resultado finais.
</p>

~~~ python
x = dataframe[["diferentesLabSemanas89", "diferentesExerSemanas67","diferentesExerSemanas45","notaProva1","qsemana1" ]].values 
y = dataframe.iloc[:, 11].values
~~~
<p align = "justify">
O código abaixo utiliza a função .add(“parâmetros”)  do objeto “classifier” que foi retirado da biblioteca do keras. Podemos ver que na função add é aquela que adiciona camadas ao código, ou seja, é a função que inicializa o processo de machine learning adicionando a camada de inputs com o parâmetro “input_dim = 5” que significa que existem cinco atributos/inputs(atributos relacionado aos dados dos alunos) e adicionando a camada escondida(hidden-layer) com o parâmetro da função “units = 8” representando a existência de 8 pesos, ou seja, 8 estruturas análogas a neurônios nesse hidden-layer.
</p>

<p align = "justify">
Os comentários mostram a possibilidade de adicionar mais camadas escondidas, porém em nosso caso não foram necessárias.
</p>

~~~ python
# camada de input e a primeira camada escondida
classifier.add(Dense( activation = 'relu', input_dim = 5, units = 8, kernel_initializer = 'uniform'))

# Secunda camada escondida
#classifier.add(Dense( activation = 'relu', units = 4, kernel_initializer = 'uniform' ))

# Terceira camada escodinda
#classifier.add(Dense( activation = 'relu', units = 6, kernel_initializer = 'uniform' ))


# camada de output
classifier.add(Dense( activation = 'sigmoid', units = 1, kernel_initializer = 'uniform'))
~~~

<p align = "justify">
Na linha final é visto a adição de mais uma camada que usa como um dos parâmetros de função “units = 1”, sendo a última camada podemos presumir que ela representa a camada de outputs e que a partir do parâmetro citado a cima(units = 1) temos que há somente uma saída referente a situação aluno, podendo ser um binário sendo um relacionado a aprovação e 0 relacionado a reprovação.
<p align = "justify">
Com o que foi apresentado já se há se formado a estrutura da rede neural. São expostos 5 valores de inputs que estão relacionados a atividades dos alunos e seus desempenhos, 8 neurônios em um hidden-layer e um output que indica a possível situação final do aluno.
</p>
## Experimentos 

<p align = "justify">
Como mostrado no código do início da aba de "código", existem cinco parâmetros/atributos específicos dos alunos que foram selecionados e avaliados pela rede neural, os parâmetros foram: nota da primeira prova(prova1), notas de questionários e exercícios das semanas iniciais(diferentesExerSemanas e qsemana1) e a nota de um dos laboratórios. O fundamento para a escolha desses valores foi primeiramente a busca por analisar a constância do aluno a partir de suas notas em questões, exercícios e laboratórios semanais iniciais, checando se o mesmo está engajado nos estudos semanais, e posteriormente analisar se o aluno absorveu todos os conteúdos vistos nas semanas antes da primeira prova passando como atributo a nota da primeira prova. Além de checar a absorção dos conteúdos, passar a nota da primeira prova serve para verificar se as questões e exercícios feitos pelo aluno não foram copiados de nenhum colega. Vale salientar que todos os recursos utilizados foram de períodos iniciais do semestre, para que seja possível utilizar a rede neural  o mais rápido possível e, assim, podendo ter tempo para pode modificar o quadro determinado pelo sistema de MLP.
<p align = "justify">
Antes de analisar os resultados obtidos(fig 4) pela rede neural na parte de treino, é necessário explicar um elemento muito importante para essa análise que a matriz de confusão. A matriz de confusão(confusion matrix) é uma matriz que apresenta a quantidade de erros e acertos de um modelos de maneira a apresentar a quantidade de valores errados em sua diagonal secundária e seus acertos na diagonal principal.
</p>
<center>
<img src = "https://www.researchgate.net/profile/Fabio_Araujo_Da_Silva/publication/323369673/figure/fig5/AS:597319787479040@1519423543307/Figura-13-Exemplo-de-uma-matriz-de-confusao.png"  width = "550"></img>
<h5>(fig 3 - quadro que explica a matriz de confusão)
</h5>
</center>

<p align = "justify">
A partir da figura 3 podemos dialogar com a matriz de confusão obtida no nosso modelo na figura 4. A classe negativa representa a reprovação do aluno e a classe positiva representa a aprovação do aluno, logo, os itens da diagonal principal foram aqueles alunos que foram devidamente destinados as suas situações finais(previsão bate com o resultado real) e já aqueles presentes na diagonal secundária foram aqueles alunos que foram julgados de maneira errônea pelo sistema(resultado real não coincide com o previsto). Na matriz de confusão da fig 4, podemos ver que existem muito mais erros relacionados a casos de alunos que foram aprovados, porém a predição teria apontado uma reprovação, isso indica uma inclinação do modelo de rede neural a reprovar mais os alunos do que aprovar em suas predições.
</p>

~~~ py
print("Matriz de Confusão:")
print(cm)
print("Taxa de acerto:")
print((cm[0,0]+cm[1,1])/len(y_test) )
print(len(y_test))
~~~
~~~py
Matriz de Confusão:
[[54  8]
 [17 56]]
Taxa de acerto:
0.8148148148148148
135
~~~

<center>
<h5>(fig 4 - resultado obtido pela fase de treino da rede neural, expondo a matriz de confusão</h5>
</center>

<p align = "justify">
Por fim, podemos verificar diante a soma da diagonal principal da matriz de confusão, armazenada na variável cm no código da figura 7, a quantidade total de acertos da rede neural na fase de testes e quando dividido pelo número total de alunos(135) é possível ver a porcentagem de acerto do modelo de ML que, no  caso, foi de 81,48% dos discentes tiveram sua situação final adivinhada pelo algoritmo.
</p>