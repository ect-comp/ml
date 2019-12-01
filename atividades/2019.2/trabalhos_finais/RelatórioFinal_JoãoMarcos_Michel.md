# Reconhecimento de Gestos em Libras

## Introdução
   O relatório a seguir tem o objetivo de mostrar ao leitor um breve histórico do processo de desenvolvimento de um algorítimo, escrito pelos alunos João Marcos e Michel Rodrigo, que é capaz de reconhecer gestos em Libras (Linguagem Brasileira de Sinais).
   O projeto foi desenvolvido como atividade final da disciplina de Tópicos Avançados em Informática na Escola de Ciência e Tecnologia da UFRN, ministrada pelo professor Orivaldo Santana.
   É importante salientar que o objetivo foi cumprido (reconhecer os gestos), porém, não é algo simples o suficiente para ser concluído com tanta rapidez. Portanto, para iniciarmos o projeto, o algorítimo reconhece atualmente apenas gestos gestos em libras que representam as vogais do nosso alfabeto, mas com um pouco de refino e um banco de dados mais robusto tem grande potencial para o reconhecimento de todo o alfabeto.

## Metodologia
   O modelo de Machine Learning utilizado no processo foi o da rede SOM (Self-Organized-Maps), que, de cordo com PINHEIRO (2019) “é uma rede neural de 2 camadas que aceita padrões de N-dimensões como entrada e os mapeia para um conjunto de neurônios de saída, o qual representa o espaço dos dados a serem agrupados”.
   A ideia principal desse modelo é fazer com que dados semelhantes fiquem próximos uns dos outros e respondam de maneira semelhante às entradas.
   O algorítmo foi desenvolvido em Python e o banco de dados está armazenado em um arquivo CSV.
   O banco de dados trata-se de um conjunto de pontos (20 de cada registro) que representam as posições dos dedos que foram capturados nas imagens.

## Desenvolvimento da Aplicação
### A ideia
   A ideia surgiu a partir de uma necessidade bem comum, como entender a Linguagem Brasileira de Sinais e participar do processo de inclusão de pessoas com deficit na oralidade, mas que já se comunicam através utilizando LIBRAS?
   Atualmente encontramos aplicativos nas Lojas de Aplicativos que conseguem, através de desenhos animados em 3D, processar linguagem natural e demonstrá-la em LIBRAS, o que facilita muita coisa. Porém, o contrário não é muito comum.
   Há uma necessidade de aplicações que realizem o processo inverso, para que a pessoa que utiliza gestos para se comunicar consiga fazê-lo esperando que seu espectador esteja entendendo.

### Produção do Banco de Dados para o Treinamento da Rede
   A partir das imagens selecionadas, utilizamos a biblioteca HandKeyPointerDetector para selecionar os pontos na imagem. Em seguida criamos um laço de repetição para selecionar cada imagem, obter os pontos e armazená-los em um arquivo CSV.
~~~ python
for i in range(nPoints):
  # confidence map of corresponding body's part.
  probMap = output[0, i, :, :]
  probMap = cv2.resize(probMap, (frameWidth, frameHeight))
  # Find global maxima of the probMap.
  minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
  if prob > threshold :
      cv2.circle(frameCopy, (int(point[0]), int(point[1])), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
      cv2.putText(frameCopy, "{}".format(i), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
  # Add the point to the list if the probability is greater than the threshold
      points.append(int(point[0]))
      points.append(int(point[1]))
  else :
      points.append(0)
      points.append(0)

cv2.imwrite('Outputs/Rock/Output-Keypoints_'+str(j)+'.jpg', frameCopy)
#cv2.imwrite('Outputs/Palma/Output-Skeleton_'+str(j)+'.jpg', frame)

print("Total time taken : {:.3f}".format(time.time() - t))

print(points)

#Criar arquivo .csv
with open('keypointsRock.csv', mode='a') as employee_file:
  employee_writer = csv.writer(employee_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
  employee_writer.writerow(points)

cv2.waitKey(0)
~~~
   Em seguida podemos ver exemplos de imagens utilizadas e os pontos encontrados pela biblioteca:
![letraasn](https://user-images.githubusercontent.com/55205574/69909834-d6c39c80-13df-11ea-912d-99dd09a0df44.jpeg)
![letraacn](https://user-images.githubusercontent.com/55205574/69909833-d6c39c80-13df-11ea-8af9-e49696f99c02.jpeg)
![letraesn](https://user-images.githubusercontent.com/55205574/69909836-d6c39c80-13df-11ea-9968-dd2e16e02c01.jpeg)
![letraecn](https://user-images.githubusercontent.com/55205574/69909835-d6c39c80-13df-11ea-89b9-12595ac4de56.jpeg)

   As saída possíveis são números que representam uma vogal: 0 para A, 1 para E, 2 para I, 3 para O e 4 para U.

### Treinamento da Rede
Os parâmetros da rede foram ajustados da seguinte forma:
~~~ python
from minisom import MiniSom
som = MiniSom(x = tamanhoXdaRede, y = tamanhoYdaRede, input_len = quantidadeCaracteristicas, sigma = 1.0, learning_rate = 0.6)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 90000)
~~~
Como é encontrado o neurônio vencedor:
~~~ python
# encontra o vencedor 
x = X[1,:]
pos = som.winner(x)
print(Y_train)
print(X.size)

#matriz de zeros para decisões para cada neurônio
MNeuronios = np.zeros((tamanhoXdaRede,tamanhoYdaRede))

MContT = np.zeros((tamanhoXdaRede,tamanhoYdaRede))

#matriz de zeros para contador de decisões para frente
MContLetraA = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
#matriz de zeros para o contador de decisões para direita 
MContLetraE = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
#matriz de zeros para o contador de decisões para esquerda 
MContLetraI = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
MContLetraO = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
MContLetraU = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
cont = 0; 
for x in X: 
  pos = som.winner(x)
  MContT[pos] += 1
  if Y_train[cont] == 0:
    MContLetraA[pos] += 1
    MNeuronios[pos] = 0
  elif Y_train[cont] == 1:
    MContLetraE[pos] += 1
    MNeuronios[pos] = 1
  elif Y_train[cont] == 2:
    MContLetraI[pos] += 1
    MNeuronios[pos] = 2
  elif Y_train[cont] == 3:
    MContLetraO[pos] += 1
    MNeuronios[pos] = 3
  elif Y_train[cont] == 4:
    MContLetraU[pos] += 1
    MNeuronios[pos] = 4

  cont= cont+1
~~~
Os resultados obtidos foram muito bons, com poucas sobreposições e podem ser analisados no gráfico a seguir:
![índice](https://user-images.githubusercontent.com/55205574/69915734-27afb100-1431-11ea-96aa-9862da95f035.png)

Em seguida, o arquivo com a rede já treinada foi exportado para suas devidas aplicações.
O algorítmo responsável pelo treinamento está disponível em: https://colab.research.google.com/drive/1Zv1jMWokf_6RU1SuhRpE24h2TXh6fzw_#scrollTo=zbxtb12oRVWu.

### Experimentos e Aplicações Realizadas
* Inicialmente, o projeto foi utilizado para controlar um sistema motorizado de pequeno porte (carrinho) com alguns gestos que consideramos ser mais bem definidos, fáceis de distinguir. Cada gesto era capaz de realizar uma ação diferente: para frente, para trás, para a direita, para a esquerda.

<img src='https://user-images.githubusercontent.com/55205574/69915821-0dc29e00-1432-11ea-87bf-7db66430f830.gif' width='100'/> <img src='https://user-images.githubusercontent.com/55205574/69915822-0e5b3480-1432-11ea-96a3-7f4482b9ae5b.gif' width='100'/> <img src='https://user-images.githubusercontent.com/55205574/69915974-e53ba380-1433-11ea-80b3-eab79cdff283.gif' width='100'/> <img src='https://user-images.githubusercontent.com/55205574/69915975-e53ba380-1433-11ea-9be0-d0a9d1f0ca32.gif' width='100'/>

   * Nesse experimento, obtivemos um bom resultado quanto a resposta do sistema ao gesto que era capturado pela câmera, no entanto o processo entre a captura da imagem, reconhecimento do gesto e envio da requisição ao carrinho demorou cerca de 3 segundos. Com alguns testes observamos que essa demora se dava apenas ao processo de captura da imagem em tempo real, pois sem ele todo o processo levava menos de 1 segundo para ser concluído.
   
* Em seguida, adaptamos o código para o processamento de gestos que representassem letras do alfabeto em LIBRAS. Os resultados também se mostraram muito bons, com exceção das letra A e E que possuem algumas características muito próximas.

<img src='https://user-images.githubusercontent.com/55205574/69916734-7e22ec80-143d-11ea-910a-87e10345069c.jpeg' width='50%'/>

## Conclusão
Durante todo o processo de desenvolvimento do projeto, os experimentos retornaram bons resultados e apresentaram respostas esperadas. Porém, alguns refinos devem ser trabalhados, pois, em alguns casos, o algorítmo confunde a letra A com a letra E por apresentarem pontos com características semelhantes.
A proposta apresentada para a solução do problema seria adicionar uma segunda camada ao processo de reconhecimento do gesto. Essa camada seria acionada apenas quando o sistema identificasse a letra A ou a letra E, como uma espécie de "tira teima".

## Referência
POSNER, Erez. HandKeyPointerDetector. Disponível em <https://github.com/erezposner/HandKeyPointDetector>. Acesso em 25 de nov. de 2019.
