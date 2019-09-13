# Identificar posição de uma pessoa através pose estimation.

## Introdução

O trabalho foi desenvolvido por Maciel Barbosa como único membro.
* Descrever o problema. 

A proposta é identificar se a pessoa de uma imagem está em pé ou sentada. Para isso precisamos obter parâmetros que descrevam os diferentes tipos de posição. Foi utilizada uma rede neural de pose estimation para, a partir das imagens, retornar os pixels de uma série de articulações da pessoa presente na mesma.

* Descrever a base de dados.

A base de dados foi obtida através de uma api que consome o modelo de pose estimation do tensowflow. Outro pequeno projeto foi usado para geras os CSVs. As imagens foram cedidas por um amigo, totalizando 152 em pé e 150 sentado.

## Metodologia 

* Explicar o modelo de _machine learning_ (ML) que você está trabalhando. 
* Explicar as etapas do treinamento e teste. 
* Caso tenha selecionado atributos, explicar a motivação para a seleção de tais atributos. 

## Códigos 

* Mostrar trechos de códigos mais importantes e explicações.  

## Experimentos 

* Descrever em detalhes os tipos de testes executados. 
* Descrever os parâmentros avaliados. 
* Explicar os resultados. 
