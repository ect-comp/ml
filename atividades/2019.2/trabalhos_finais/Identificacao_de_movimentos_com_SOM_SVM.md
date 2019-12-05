# Identificação de letras usando Multilayer Perceptron  (MLP)

## Introdução

A base de dados utilizada para esse trabalho foi extraida de 222 imagens que passaram por um programa para gerar o arquivo csv. Esse programa
apresenta várias imagens a um servidor que contém o detector de posições de juntas PoseNET que retorna 17 pontos de coordenadas X e Y para cada
imagem apresentada. Gerando esse vetor de dados das 222 imagens apresentadas e adicionando a classificação de cada imagens (se perna levantada, esqueda ou direita, ou braço, esquerdo ou direito, levantado) formamos a base de dados que será utilizada para o projeto. O objetivo é gerar com a rede Self-Organized Maps (SOM) um mapa de característas que indique em que neurônios estão situados cada posição classificada no arquivo csv apresentado a rede. Após isso apresentar um vetor de "imagens" (já processadas pelo poseNET) e gerar um vetor de ativação dos neurônios.

## Metodologia 

O modelo de machine learning utilizado para a clusterização dos dados de submissões foi a SOM, self-organizing map. Consiste de uma rede neural não supervisionada que é responsável por criar um mapa dimensionado que descretiza a base de dados e a organiza de forma adequada. A SOM conta com um treinamento a partir de valores de entrada escritos na base de dados. Após o processo de treinamento, ele passará por uma fase de mapeamento, a qual será responsável por classificar de forma automática a base de dados em neurônios e organizando esses neurônios. O tamanho configurado foi de 16 por 16.
Ja SVM é um modelo de machine learning que consegue aplicar uma tranformação no espaço de dados dessa forma conseguindo separar dados que aparentemente não aparentam ser não lineramente separáveis, mas que em algum plano do espaço se tornam linearmente separáveis. Ela utiliza funções de Kernel, em SVMs não lineares, para calcular hiperplanos ótimos que consigam separar da melhor forma os dados fornecidos no processo de treinamento. 

## Códigos 

```python
 # encontra o vencedor 
x = X_train[1,:]
pos = som.winner(x)

# matriz de zeros para contador de aprovados 
MCont1 = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
# matriz de zeros para contador de aprovados 
MCont2 = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
# matriz de zeros para contador de aprovados 
MCont3 = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
# matriz de zeros para contador de aprovados 
MCont4 = np.zeros((tamanhoXdaRede,tamanhoYdaRede))

# matriz de zeros para o contador de reprovados 
MContT = np.zeros((tamanhoXdaRede,tamanhoYdaRede))
cont = 0; 
for x in X_train: 
  pos = som.winner(x)
  if (Y_train[cont] == 'braco_direito'):
    MCont1[pos] += 1
  elif (Y_train[cont] == 'braco_esquerdo'):
    MCont2[pos] += 1
  elif (Y_train[cont] == 'perna_direita'):
    MCont3[pos] += 1
  elif (Y_train[cont] == 'perna_esquerda'):
    MCont4[pos] += 1
  MContT[pos] += 1
  cont= cont+1
```
criando as matrizes separado cada classe
```python
cont = 1;
for i in range(len(MContT)):
  for j in range(len(MContT)):
    plt.subplot(tamanhoXdaRede,tamanhoYdaRede,cont)
    cont=cont+1
    sizes = [MCont1[i][j], MCont2[i][j], MCont3[i][j], MCont4[i][j]]
    plt.pie(sizes)
plt.show()
```
criando uma matriz de gráficos de pizza para ver a separação de cada classe

projeto disponivel em https://colab.research.google.com/drive/1OqV-T_qhFxDsS2gVX58L3dhi6N8YDAYh
 
