# Reconhecimento Facial

## Introdução
O sistema de reconhecimento facial foi feito utilizando a "CNN", a qual significa "Convolutional Neural Network", que é responsável por pegar a imagem, processá-la e a classificar em determinadas categorias.
O trabalho desenvolvido utilizou 15 fotos de um dos colaboradores e possuiu, como resposta, seu reconhecimento ao longo de diversas outras fotos.

### Contribuidores
- Leonardo Queiroz, aluno de bacharelado em engenharia mecânica na Universidade Federal do Rio Grande do Norte, contato em:
* [Github](https://github.com/leocqueiroz)

- Gabriel Varela, aluno da Universidade Federal do Rio Grande do Norte do curso de bacharelado em engenharia mecatrônica. Contato em:
* [Github](https://github.com/gabrielvrl)
* [Site](https://gabrielvrl.github.io/)

## Metodologia
O trabalho de Reconhecimento Facial com Python e OpenCV foi desenvolvido dentro da plataforma do Google o Google Colaboratory.
O primeiro passo para o trabalho foi instalar as bibliotecas necessárias para o desenvolvimento do trabalho

```py
!pip3 install CMake
!pip3 install dlib==19.10.0
!pip3 install face_recognition
!pip3 install imutils
!pip3 install opencv-python
!pip3 install matplotlib
```

Fornecemos permissão a nossa pasta do Google Drive para o treinamento

```py
from google.colab import drive
drive.mount('/content/gdrive')
baseDrive = '/content/gdrive/My Drive/'
```

O próximo passo é importar as bibliotecas necessárias

```py
from imutils import paths
import face_recognition
import argparse
import pickle
import cv2
import os
```

Fornecemos o caminho das imagens de treinamento

```py
caminhoImagens = baseDrive + 'Reconhecimento-Facial/Gabriel'
print("Quantificando faces ...")
imagePaths = list(paths.list_images(caminhoImagens))
print("Faces quantificadas com sucesso: "+str(len(imagePaths)))
```

Podemos inicializar as listas de codificações e nomes conhecidos

```py
knownEncodings = []
knownNames = []
```

Fazemos um loop em cada imagem do diretório

```py
for (i, imagePath) in enumerate(imagePaths):
	# Extrair o nome da pessoa do caminho da imagem
	print("PROCESSANDO IMAGEM {}/{}".format(i + 1,
		len(imagePaths)))
	name = imagePath.split(os.path.sep)[-2]

	# Carregar a imagem de entrada e convertê-la de RGB (ordem OpenCV)
	# Para ordenar dlib (RGB)
	image = cv2.imread(imagePath)
	rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	# Detectar as coordenadas (x, y) das caixas delimitadoras
	# Correspondente a cada face na imagem de entrada
  # Sistemas de detecção: cnn ou hog.
	boxes = face_recognition.face_locations(rgb,
		model='cnn')
  
	# Calcular a incorporação facial para o rosto, caracteristicas.
	encodings = face_recognition.face_encodings(rgb, boxes)

	# loop sobre as codificações
	for encoding in encodings:
		# adicione cada codificação + nome ao nosso conjunto de nomes conhecidos e codificados
		knownEncodings.append(encoding)
		knownNames.append(name)
    
print('IMAGENS PROCESSADAS COM SUCESSO!')
```

Salvamos nosso arquivo de treino para podermos utiliza-lo

```py
nomeEncode = 'treino-apresentacao.pickle'
print("Codificações de serialização")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(baseDrive + 'Reconhecimento-Facial/' + nomeEncode, "wb")
f.write(pickle.dumps(data))
f.close()
print(nomeEncode, "salvo com sucesso!")
```

Podemos carregar o arquivo de treino

```py
# Carregar os rostos e os encaixes conhecidos
print("Carregando")
data = pickle.loads(open(baseDrive + 'Reconhecimento-Facial/' + nomeEncode, "rb").read())
print('Treinamento carregado com sucesso')
```
Carregar a imagem e convertendo-a de BGR para RGB

```py
imagemEntrada = baseDrive + 'Reconhecimento-Facial/examples/professor.jpeg'
image = cv2.imread(imagemEntrada)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print("Imagem carregada com sucesso")
```

Detectamos as coordenadas (x,y) das caixas delimitadoras correspondentes e para cada face na imagem de entrada e, em seguida, calcula-se os encaixes faciais para cada rosto. Incializamos também a lista de nomes para cada rosto detectado

```py
print("Identificando faces")
boxes = face_recognition.face_locations(rgb,
	model='cnn')
# Model pode ser cnn ou hog
encodings = face_recognition.face_encodings(rgb, boxes)
print("Faces detectadas")

names = []
```

Laço sobre cada quadro

```py
for encoding in encodings:

  print("Processando informações")

  # tentar combinar cada rosto na imagem de entrada para o nosso conhecido
  matches = face_recognition.compare_faces(data["encodings"], encoding)

  name = "Desconhecido"

  # verifique se encontramos uma correspondência
  if True in matches:
    # encontrar os índices de todos os rostos combinados, em seguida, inicializar um
    # dicionário para contar o número total de vezes que cada rosto
    # foi correspondido
    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
    counts = {}

    # fazer um for sobre os índices correspondentes e manter uma contagem para
    # cada face do rosto reconhecida
    for i in matchedIdxs:
      name = data["names"][i]
      counts[name] = counts.get(name, 0) + 1

    # determinar a face reconhecida com o maior número de
    # votos (nota: no caso de um improvável empate, o Python
    # selecione a primeira entrada no dicionário)
    name = max(counts, key=counts.get)

  # atualizar a lista de nomes
  names.append(name)
  ```

Loop sobre as faces reconhecidas

```py
for ((top, right, bottom, left), name) in zip(boxes, names):
  # Desenhe o nome da face prevista na imagem
  cv2.rectangle(rgb, (left, top), (right, bottom), (200, 0, 0), 2)
  y = top - 15 if top - 15 > 15 else top + 15
  cv2.putText(rgb, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 0), 2)
```

Plot da imagem e arquivo salvo

```py
import sys
import io
from PIL import Image
os.getcwd()
Image.fromarray(rgb).save('leo2.jpg')
#cv2.imwrite("leozinh.png",rgb)

from matplotlib import pyplot as plt
plt.imshow(rgb)
#plt.grid(None)
plt.title('Foto com reconhecimento')
#fig = plt.gcf()
#plt.show()
#fig.savefig('teste.png', format='png')
#plt.savefig('nomeDaFigura.png') 
#plt.savefig('reconhecimento.png', transparent = True)
plt.show()
```

## Resultados
As imagens abaixo mostram o reconhecimento do rosto treinado, e identifica os demais como rostos desconhecidos.

![foto1](https://github.com/leocqueiroz/MachineLearning/blob/master/FaceRecognition/Fotos/foto1.PNG)

![foto2](https://github.com/leocqueiroz/MachineLearning/blob/master/FaceRecognition/Fotos/foto2.PNG)

![foto3](https://github.com/leocqueiroz/MachineLearning/blob/master/FaceRecognition/Fotos/foto3.PNG)

![foto4](https://github.com/leocqueiroz/MachineLearning/blob/master/FaceRecognition/Fotos/foto4.PNG)

Para visualizar o código, basta acessar o link abaixo:
https://colab.research.google.com/drive/1qQzSlR_g0nJOF1PaE51jXadLt3aHv3XA