# Reconhecimento de caracter escrito à mão 
#### JORGE LUÍS GURGEL FERNANDES <br>
#### SEVERINO MIGUEL DO NASCIMENTO NETO <br><br>
 
## Introdução
<p align="justify">
O presente relatório tem como propósito, descrever o desenvolvimento de um programa que reconhece um caracter escrito a mão por meio de uma rede neural.  
 
## Metodologia<br> 
<p align="justify">
Para cumprir com o nosso objetivo, utilizamos a linguagem Python e, através dela, a rede neural Multilayer Perceptron(MLP), em conjunto com a base de dados que foi disponibilizada em sala de aula.  
 
## Base de dados
A base de dados consiste em um conjunto de arquivos de imagens com caracteres escritos à mão, este serão usados como ferramenta de treinamento para o funcionamento do código.
<br>

## Passo à passo:<br>
 
### Trechos principais:
Trecho em que foi inserida as devidas bibliotecas e base de dados manipulada:
<br>

```py	
import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt 
from keras.models import Sequential 
from keras.layers import Dense
from keras.optimizers import Adam 
from keras.utils.np_utils import to_categorical
```
<br>

Após descompactar os arquivos foi utilizado uma instrução com a finalidade de identificar as imagens [images] e os rótulos [labels], armazenado em duas matrizes

```py	
images = []
labels = []
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
``` 
 <br>

 Após isso, o código manipula as imagens deixando-as em tons de cinza e inverte as suas cores, com o propósito de facilitar a identificação do caracter, para inseri-los nos vetores de imagens e rótulos.
 <br>

```py
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
<br>

Na trecho abaixo selecionamos o valor de 25% para teste, restando 75% para o treinamento.

```py
X_train, X_test, y_train, y_test = train_test_split(images, labels_enc, test_size=0.25, random_state=1)
```

<img src="https://github.com/migueloten/imagens/blob/master/Sem%20t%C3%ADtulo.png?raw=true">

<br>

Depois manipulamos os parâmetros do treinamento com a finalidade de obter uma maior acurácia.

```py
history = model.fit(X_train, y_train, validation_split=0.1, epochs=250, batch_size = 80, verbose = 1, shuffle = 15)
```

Finalmente obtivemos uma acurácia de 61%.

## Teste 

Para realizar o teste com o código desenvolvido, carregamos uma imagem aleatória de um caracter escrito à mão para que o código faça o tratamento da imagem e reconhecimento desse caracter.

### Exemplo:

carregamento da imagem:
<br>
<img src="https://github.com/migueloten/imagens/blob/master/teste1.png?raw=true">

imagem tratada e predição:
<br>
<img src="https://github.com/migueloten/imagens/blob/master/teste2.png?raw=true">



## Conclusão<br>

<p align="justify">
Percebe-se que uma acurácia de 61% não é suficientemente bom, dado que poucos teste obtiveram sucesso, mas isso se deu devido ao banco de dados insuficiente e falho, para um processo com essa complexidade, seria necessário uma base de dados maior e consequentemente mais completa.

</p>
 
