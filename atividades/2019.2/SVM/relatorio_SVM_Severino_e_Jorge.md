# USO DE SVM PARA PREVISÃO DE POSIÇÃO
 
## Introdução
<p align="justify">O grupo é composto por Jorge Luís e Severino Miguel, alunos do curso de bacharelado em Ciências e Tecnologia da UFRN.
O presente relatório trás uma abordagem dos métodos e passos para desenvolver um algorítmo, que utiliza uma base de dados gerada pelo grupo, 
que possa predizer o estado de uma pessoa(levantado ou sentado).</p>

## Metodologia

### Primeira etapa: Coleta dos dados

<p align="justify">Primeiramente nós gravamos um vídeo em que um dos integrantes do grupo se posicionava nos dois estados(sentado e levantado).
Depois, transformamos esse vídeo em várias imagens no formato .jpg, de onde extraímos uma base de dados com a ajuda da API Pose estimation.<p>
 
 ![](new.gif)
<img style="float:left;" src="https://github.com/migueloten/relatorio_svm/blob/master/database/new.gif" width="250" height="250">

#### Códigos<br>

```py

#Definindo o caminho das fotos
gitraw = "https://raw.githubusercontent.com/"
user = "migueloten/"
repository = "relatorio_svm/"
branch = "master/"
folder = "database/img"
ext = ".jpg"
arrImages = []
arrPos = []

#Colocando as fotos em um array
for i in range(1,53):
  arrImages.append(gitraw+user+repository+branch+folder+str(i)+ext)

#Mapeando as fotos e salvando os dados em um array
r = req.get("http://poseestimation.herokuapp.com/header")
header = r.text.replace('[', '').replace(']', '')+",estado"

for i in range(52):
  r = req.get("http://poseestimation.herokuapp.com/estimate?url="+arrImages[i])
  arrPos.append(str(i+1)+','+r.text.replace('[', '').replace(']', ''))

#Definindo os estados
for i in range(52):
  if i<27:
    arrPos[i]+=",levantado"
  else:
    arrPos[i]+=",sentado"

#Exportando os dados em .csv
f= open("dataBase.csv","w+")
f.write(header+'\r\n')
for i in range(52):
  f.write(arrPos[i]+'\r\n')
f.close()  

```

### Segunda etapa: Análise dos dados

<p align="justify">Após a coleta dos dados, fizemos a manipulação através da ferramenta Support Vector Machine(SVM), 
separamos 25% dos dados para teste e colocamos o restante para o treino.
Selecionamos os atributos que vão influenciar nas decisões da nossa rede, assim como o que a nossa rede vai retornar, 
também manipulamos os parâmetros do SVM(gamma e C), para que a rede funcione de uma forma mais otimizada e por fim
obtemos uma matriz de confusão.


#### Códigos<br>
 

```py	
#Selecionando a base de dados
FILE_TO_DOWNLOAD =  "dataBase.csv"
DOWNLOAD_ROOT = "https://github.com/migueloten/relatorio_svm/raw/master/"
DATA_PATH = "csv/"
DATA_URL = DOWNLOAD_ROOT + DATA_PATH + FILE_TO_DOWNLOAD
 
def fetch_data(data_url="https://raw.githubusercontent.com/migueloten/relatorio_svm/master/csv/dataBase.csv", data_path="csv/", file_to_download="dataBase.csv"):
  if not os.path.isdir(data_path):
    os.makedirs(data_path)
  urllib.request.urlretrieve(data_url, data_path+file_to_download)
 
fetch_data()
 ```
 
 
```py
#Atributos escolhidos da base de dados
X = dataset.iloc[:,2:35].values
y = dataset.iloc[:,35].values
 ```
 
```py
#Manipulando os parâmetros do SVM (gamma e C)
#classifier = SVC(kernel = 'linear', random_state = 0)
classifier = Pipeline((
("scaler", StandardScaler()),
("svm_clf", SVC(kernel="rbf,", gamma=0.1, C=10))
)) 
```
 
```py
#Descobrindo a matriz de confusão
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)
``` 

### Terceira etapa: Interpretação dos resultados

##### Matriz de confusão
<p align="justify">É um tabela que mostra as frequências de classificação para cada classe do modelo.
Cada uma das quatro coordenadas da matriz tem um significado, que podem ser:</p>

* Verdadeiro positivo (true positive — TP): representa as vezes que o valor foi positivo, como esperado.
* Falso positivo (false positive — FP): representa as vezes que o resultado deu positivo, mas o esperado era que desse negativo.
* Falso verdadeiro (true negative — TN): representa as vezes que o resultado foi negativo, como esperado.
* Falso negativo (false negative — FN): representa as vezes que o resultado deu negativo, mas o esperado era que desse positivo.
<br><br>

|                          | valor encontrado positivo | valor encontrado negativo |
|:------------------------:|:-------------------------:|:-------------------------:|
|<b>valor real positivo</b>|            TP             |             FP            | 
|<b>valor real negativo</b>|            FN             |             TN            |


##### Nossa Matriz de confusão:

<br>

|                | levantado | sentado |
|:---------------|:---------:|:-------:|
|<b>levantado</b>|     7     |    0    | 
|<b>sentado</b>  |     0     |    6    | 

<p align="justify">Por fim, utilizando nossa matriz de confusão, podemos medir a acuracia do nosso experimento atravéz de uma equação simples:</p>

#### Acuracia = (TP + TN)/(TP + FP + FN + TN)
<p align="justify">Medindo a nossa acuracia encontramos 100% de acerto dos nossos testes, o que mostra que tivemos bons resultados diante dos dados que pudemos fornecer.</p>

 #### Link para código fonte:<br>
<a href="https://colab.research.google.com/drive/1rDdSCldgDo7eVW3qpJv2bLlNuAcICZ0I#scrollTo=_HE3vh8Hn7uS">Clique aqui</a> <br><br>
 
 
## Experimentos<br>
 
### Parâmetros manipulados:
 
* nose_x, nose_y:  Posição do nariz
* leftEye_x, leftEye_y	: Posição do olho<br>
* rightEye_x, rightEye_y: Posição do olho direito<br>
* leftEar_x, leftEar_y: Posição da orelha esquerda<br>
* rightEar_x, rightEar_y: Posição da orelha direita	<br>
* leftShoulder_x, leftShoulder_y: Posição do ombro esquerdo<br>
* rightShoulder_x, rightShoulder_y	: Posição do ombro direito<br>
* leftElbow_x, leftElbow_y: Posição do cotovelo esquerdo<br>
* rightElbow_x, rightElbow_y: Posição do cotovelo direito	<br>
* leftWrist_x, leftWrist_y: Posição do pulso esquerdo<br>
* rightWrist_x, rightWrist_y: Posição do pulso direito<br>
* leftHip_x, leftHip_y: Posição do quadril esquerdo<br>
* rightHip_x, rightHip_y: Posição do quadril direito<br>
* leftKnee_x, leftKnee_y: Posição do joelho esquerdo
* rightKnee_x, rightKnee_y: Posição do joelho direito
* leftAnkle_x, leftAnkle_y: Posição do tornozelo esquerdo
* rightAnkle_x, rightAnkle_y: Posição do tornozelo direito
 
<p align="justify">Por fim, analisando a nossa matriz de confusão e a acuracia, que apresentou uma taxa de 100% de acerto, 
podemos ver que foi possivel predizer o estado de um corpo analisando as partes de seu corpo através do codigo implementado</p>
 
