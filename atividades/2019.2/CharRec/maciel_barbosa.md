# Reconhecimento de Letras com CNN
## Introdução

A equipe é composta apenas por mim.  
O objetivo é identificar a letra presente em uma imagem. Para isso foi utilizada uma base de dados composta por 127
exemplos de cada letra, totalizando 3302 imagens.   

## Metodologia 

O desevolvimento da atividade pode ser dividida em 3 etapas: preparação dos dados, treinamento e teste. Na preparação,
a base de dados é convertida para um formato adequado para o treinamento. No treinamento é definida a configuração da CNN e feita
a compilação do modelo.
No teste é verificado a saída de alguns exemplos.

## Códigos 

Definição do modelo:
```
model = Sequential()
model.add(Conv2D(32, (5,5), input_shape=(28,32,1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
```

## Experimentos 

Os resultados dos testes não foram satisfatórios. Acredito que seria mais bem sucedido caso houvesse uma melhor distribuição dos
datasets de validação para cada tipo de letra.
### Base de dados 
* Caracteres escritos a mão, https://drive.google.com/open?id=1B4fy3Nqy9AoEnC2MuuPaCstcrJ-FZstW
 
