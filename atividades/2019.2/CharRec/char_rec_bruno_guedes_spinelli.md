# Reconhecimento de caracteres escritos a mão utilizando Multi-layer Perceptron (MLP)

## Introdução
Este trabalho foi realizado por Bruno Guedes Spinelli, aluno do curso de Engenharia Biomédica da UFRN.

O objetivo deste trabalho é desenvolver um algoritmo de  rede neural artificial convolucional do tipo Multi-layer Perceptron (MLP) capaz de reconhecer e identificar caracteres escritos a mão. Para o treinamento da rede foi utilizada um banco de imagens contendo diversas imagens de cada letra do alfabeto (de A a Z, totalizando 26 letras) e para o teste foram utilizadas imagens contando caracteres feitos a mão em um programa de edição de imagem.

## Metodologia

O Perceptron foi o primeiro modelo de rede neural artificial criado com a capacidade de aprendizado. Sendo baseado no funcionamento dos neurônios e sinapses biológicas, o Perceptron é capaz de classificar as entradas, de forma binária, em dois grupos distintos e sua arquitetura é baseada em 3 elementos principais:

1) Uma camada de entrada (X).
2) Uma camada de saída (Y).
3) Um conjunto de pesos (W) ligando a entrada à saída.

![perceptron]

**Figura 1 - Representação de um Perceptron**

Neste tipo de rede a aprendizagem da rede é realizada pela modificação dos pesos (W) durante o seu treinamento. Entretanto, o Perceptron apenas é capaz de solucionar problemas linearmente separáveis, o que é um grande problema afinal problemas reais normalmente não podem ser solucionados apenas com uma reta para separar seus elementos. Por este motivo foi escolhido uma rede neural Multi-Layer Perceptron (MLP) para a realização dessa atividade.

O MLP é uma rede neural artificial muito parecida com o Perceptron, porém possui mais de uma camada de neurônios e pode ser utilizada para situações que exijam mais de uma reta para a separação correta dos elementos de entrada. Sua arquitetura consiste nos seguintes elementos:

1) Uma camada de entrada (X).
2) Uma ou mais camadas escondidas (Hidden layers).
3) uma camada de saída (Y).

![mlp]

**Figura 2 - Representação de um MLP**

O principio de funcionamento da MLP é dado pela propagação do sinal de entrada por todas as camadas da rede até que chegue a saída, ou seja, os neurônios das camadas subsequentes utilizam como entrasa o sinal de saída dos neurônios das camadas anteriores, seguida do calculo do erro, utilizando as saídas produzidas pelos neurônios da última camada, e da correção dos pesos de todos os neurônios, a partir da última camada, minimizando seus erros. Desta forma a cada treinamento realizado a rede neural tende a aumentar sua chance de acerto.

Uma rede neural convolucional (CNN do inglês Convolutional Neural network ou ConvNet) é uma classe de rede neural artificial que vem sendo aplicada com sucesso no processamento e análise de imagens digitais. Em uma CNN os dados, geralmente imagens, são pré-processados de modo que se evidencie as características desejadas, logo após, os dados são descritos em um vetor de características que é usado como entrada de treinamento da rede,  o que auxilia a rede neural a ter uma melhor aprendizagem.

## Códigos 

A rede neural foi codificada utilizando a linguagem de programação Python. Primeiramente foi realizada a descompactação do aquivo contendo as imagens:

```py
!unzip letras.zip
```

Em seguida as funções ```traverse_dir``` e ```extract_data```  foram utilizadas para identificar as imagens e os rótulos contidos no arquivo descompactado, e então salva-los. Enquanto a função ```read_image``` foi utilizada para alterar as imagens para tons de cinza:

```py
def traverse_dir(path):
    for file_or_dir in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file_or_dir))
        print(abs_path)
        if os.path.isdir(abs_path):  # dir
            traverse_dir(abs_path)
        else:                        # file
            if file_or_dir.endswith('.jpg'):
                image = read_image(abs_path)
                images.append(image)
                labels.append(path[len(path)-1])

    return images, labels
    
    def read_image(file_path):
    image = cv2.imread(file_path)
    # converte para tons de cinza 
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # inverte a cor 
    image = cv2.bitwise_not(gray_scale) 
    return image


def extract_data(path):
    images, labels = traverse_dir(path)
    images = np.array(images)

    return images, labels
```

Logo após, os dados foram divididos em dados de teste e dados de treinamento e foi definido o modelo de treinamento da rede que seria usado:

```py
X_train, X_test, y_train, y_test = train_test_split(images, labels_enc, test_size=0.3, random_state=1)

def leNet_model():
  model = Sequential()
  model.add(Conv2D(30, (5,5), input_shape=(28,32,1), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Conv2D(15,(3,3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(500, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  return model
```
Por fim foi realizado o treinamento da rede, em que se obteve um valor de 83% de acurácia:

```py
history = model.fit(X_train, y_train, validation_split=0.1, epochs=50, batch_size = 150, verbose = 1, shuffle = 1)
```

## Experimentos

Os testes da rede foram realizados com imagens de letras escritas a mão em um editor de imagens digital. As imagens tiveram seus tamanhos igualados aos tamanhos das imagens de treinamento da rede, convertidas para tons de cinza e por fim expostas a rede, que retornou a sua resposta para qual letra estaria escrita na imagem, conforme pode ser vistos abaixo:

* **Letra A:**

![a]

**Figura 3 - Resultado da imagem contando a letra A**

* **Letra B:**

![b]

**Figura 4 - Resultado da imagem contando a letra B**

* **Letra G:**

![g]

**Figura 5 - Resultado da imagem contando a letra G**

* **Letra H:**

![h]

**Figura 5 - Resultado da imagem contando a letra H**

* **Letra T:**

![t]

**Figura 6 - Resultado da imagem contando a letra T**

* **Letra W:**

![w]

**Figura 7 - Resultado da imagem contando a letra W**

Após verificar e analisar os resultados obtidos, podemos perceber que apesar de apresentar uma acurácia relativamente alta (83%), quando testada com os dados de teste do bando de imagens original, ao ser testada com imagens de outras fontes a rede neural não consegue se mostrar eficiente em predizer a letra apresentada nas imagens. Esse ocorrido provavelmente se deve ao fato do bando de imagens, com o qual a rede foi treinada, não ser grande o suficiente para representar de forma real 26 classes diferentes de letras, causando confusão ao aprendizado da rede neural.

<!-- Links -->

[perceptron]: https://upload.wikimedia.org/wikipedia/commons/thumb/b/b4/Sadssa.png/469px-Sadssa.png
[mlp]: https://1.bp.blogspot.com/-Xal8aZ5MDL8/WlJm8dh1J9I/AAAAAAAAAo4/uCj6tt4T3T0HHUY4uexNuq2BXTUwcChqACLcBGAs/s1600/Multilayer-Perceptron.jpg
[a]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/resA.png
[b]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/resB.png
[g]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/resG.png
[h]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/resH.png
[t]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/resT.png
[w]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/resW.png
