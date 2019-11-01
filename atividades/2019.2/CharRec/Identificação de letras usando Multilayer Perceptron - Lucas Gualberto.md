# Identificação de letras usando Multilayer Perceptron  (MLP)

## Introdução

A base de dados utilizada para esse trabalho é composta de 117 imagens de cada letra do alfabeto da língua portuguesa.
O objetivo principal de utilização dessa base de dados é identificar, dada uma letra qualquer, que letra do alfabeto
a imagem fornecida contém.

## Metodologia 

O modelo de machine learning (ML) usado para a classificação das letras foi o MLP. Para entender melhor o funcionamento desse modelo é impressindível aprender sobre a ML da qual ele é derivado, o Perceptron. Esse funciona de uma maneira semelhante a um neurônio, onde os dentritos recebem o estimulos elétrico, esses estímulos são processados pelo núcleo do neurônio e depois transmitidos através dos axônios para os dentritos de outra célula nervosa. A sinapse que é a passagem dessa informação de uma célula para outra por meio dos neurotransmissores é regulada pela quantidade desses neurotransmissores presentes em cada terminação dos neurônios. Tal funcionamento é abstraido para o Perceptron em forma de pesos (neurotransmissores) que regulam unidades (neurônios) que podem possuir diferentes funções de ativação. O Multilayer Perceptron difere da Perceptron quanto ao número de camadas existentes. No Perceptron existem apenas 2 camadas, onde uma é de entrada e outra é de saída. Já na MLP além das camadas de entrada e saída existem camadas escondidas que podem tem números variados de unidades. A MLP usada para esse trabalho foi configurada com 4 camadas, sendo a de entrada uma camada de convolução2D que passa 30 filtros de 5 por 5 numa matriz de 28 por 32 por 1 e ativação 'relu'. A segunda camada também de revolução de 15 filtros de 3 por 3 e ativação 'relu'. A terceira camada e a única camada escondida utilizada para esse trabalho foi composta de 500 neurônios com ativação 'relu'. Por fim a última camada com 26 camadas e ativação 'softmax'.

## Códigos 

* Mostrar trechos de códigos mais importantes e explicações.  

## Experimentos 

* Descrever em detalhes os tipos de testes executados. 
* Descrever os parâmentros avaliados. 
* Explicar os resultados. 

 
