# Identificar posição de uma pessoa através pose estimation.

## Introdução

O trabalho foi desenvolvido por Maciel Barbosa como único membro.

A proposta é identificar se a pessoa de uma imagem está em pé ou sentada. Para isso precisamos obter parâmetros que descrevam os diferentes tipos de posição. Foi utilizada uma rede neural de pose estimation para, a partir das imagens, retornar os pixels de uma série de articulações da pessoa presente na mesma.

A base de dados foi obtida através de uma api que consome o modelo de pose estimation do tensowflow. Outro pequeno projeto foi usado para geras os CSVs com esses dados. As imagens foram cedidas por um amigo, totalizando 152 em pé e 150 sentado.

## Metodologia  

O modelo utilizado foi o Support Vector Machine (SVM). Trata-se de um algoritmo de aprendizado supervisionado que utiliza uma função kernel para mapear um conjunto de pontos de dados. A limite de decisão é representado por um hiperplano, que faz a separação dos dados em classes.

Os dados gerados na api exportado em csv é importado através do pandas. A seguir, o valores e os labels são extraídos do dataset, sendo definido 25% para teste. Após isso, os dados são normalizados para o mapeamento e o treinamento é de fato realizado.

Obtei por manter todos os parâmentros gerados na api do pose estimation. Acredito que os dados em excesso não influenciaria no resultado final.

## Códigos 
  
Utilizei o código exemplo disponibilizado pelo professor, fazendo algumas adequações.

```
X = dataset.iloc[:,2:33].values
y = dataset.iloc[:,34].values
```
Nesse trecho é feita a divisão dos dados exportados do csv. O `X` são os dados. Já o `y` lista a categoria na qual cada dado de `X` se encaixa. Nesse projeto as categorias (labels) são `sentado` e `em-pe`.

```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

Acima, é feita a separação entre dados de treinamento e de teste. O teste contém 25% do total dos dados.

```
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

Padronização dos dados através de uma função kernel. Nesse momento é feito uma adequação que possibilita a separação das classes.

```
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)
```

Agora, o treinamento é executado. Este encontra o hiperplano que melhor separa os dados.

## Experimentos 

```
y_pred = classifier.predict(X_test)
print(y_test[0:35])
print(y_pred[0:35])
```

Comparando os dados do grupo de teste com as respectivos resultados do treinamento é possível perceber uma boa margem de acerto.

```
[[38  0]
 [ 5 33]]
```

A matriz de confusão nos revela que a rede confunde algumas imagens `sentado` como `em-pe`.
