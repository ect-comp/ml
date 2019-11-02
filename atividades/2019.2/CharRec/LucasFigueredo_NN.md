# Identificação de caracteres com Deep Learning

## Introdução
A equipe de desenvolvimento é formada por Lucas Figueredo Varela Alves , discente da disciplina de Tópicos Avançados em Informática I, ministrada pelo docente Orivaldo Vieira de Santana.

A segmentação de caracteres e a identificação destes em um texto é uma abordagem do uso da Visão Computacional para os mais diversos fins. Por exemplo, pode-se pensar na correção de uma prova, objetiva, a partir da identificação das alternativas assinaladas pelo estudante. Essa etapa é como um preparativo para aplicações ainda maiores, como a interpretação de linguagem natural, área a qual aborda a compreensão de blocos de texto por meio de métodos computacionais de inteligência artificial.

No caso do presente relatório, foca-se na perspectiva de identificar as letras do alfabeto com o uso de Redes Convolucionais. A base de dados utilizada foi concebida pelo docente da disciplina e nela constam mais de 80 imagens de cada letra.

As redes neurais de aprendizado profundo (Deep Learning) podem ser utilizadas para classificar imagens, carros autônomos, chatbots, assistentes etc. Para compreendermos melhor como elas funcionam na aplicação que será abordada, vale-se abordar um pouco sobre como uma imagem é entendida pelo computador. As imagens preto e brancas necessitam de apenas um canal para ser representada, como mostrado na Figura 1. Os valores de cada pixel varias de 0 (preto) ao número 255 (branco), no caso das coloridas, faz-se necessário o uso de três canais, como os RGB (vermelho, verde e azul), como pode ser visto na Figura 2.

<center>
<img src="https://miro.medium.com/max/543/1*8Ja5x9y-7-Mecxth7_zQOw.jpeg"/>
<h5>Figura 1: Visualização da matriz representativa de uma imagem preto e branco.
Fonte: Data Hackers </h5></center>

<center>
<img src="https://miro.medium.com/max/1546/1*wtEvRZ5wsupHtJ6NDaaUmg.png"/>
<h5>Figura 2: Representação dos três canais para imagens coloridas.
Fonte: Data Hackers </h5></center>

## Metodologia

Agora discutindo um pouco sobre essas redes, em linhas gerais, são redes neurais artificiais de maior escala. Esse conceito existe há mais de 50 anos, mas só se popularizou há pouco tempo. Isso se deve ao processamento computacional requerido por redes grandes, que é uma característica que impacta diretamente na eficiência da rede construída. Na figura 3, há um comparativo da performance de diferentes redes neurais no que diz respeito a quantidade de camadas e neurônios.
<center>
<img src="https://lilianweng.github.io/lil-log/assets/images/data_size_vs_model_performance.png"/>
<h5>Figura 3: Relação entre performance e tamanho da rede neural.
Fonte: Lil'Log </h5></center>
Portanto, os principais parâmetros a serem definidos são os relacionados a quantidade de camadas escondidas e de neurônios para o modelo.

Em linhas gerais, a definição de uma rede desse tipo está atrelada à representação capaz de modelar relações complexas entre dados. A maioria dos modelos presentes nessa abordagem são não-supervisionados.

A abordagem trazida neste relatório se assemelha ao supracitado, como as etapas se sucederam está descrito abaixo.

## Códigos

A extração de caracteres foi feita a partir de um código fornecido pelo docente. Minados das imagens de cada caractere, o primeiro passo é definir quantas classes serão identificadas  e as configurações básicas da rede:

```py
num_classes = 26
def create_model():
  model = Sequential()
  model.add(Dense(250, input_dim=num_pixels, activation='relu'))
  model.add(Dense(100, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(num_classes, activation='softmax'))
  model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
  return model
```
No código acima, define-se que a camada de entrada na rede terá 250  neurônios e as demais, respectivamente, 100, 50 e 26.

```py
model = create_model()
print(model.summary())
```
Logo em seguida, parte-se para a criação do modelo e a visualização das suas características gerais, podendo, assim, visualizar informações sobre as camadas geradas.

```py
history = model.fit(X_train, y_train, validation_split=0.1, epochs=300, batch_size = 250, verbose = 1, shuffle = 15)
```

Com o modelo configurado, basta-se treiná-lo com os dados da nossa aplicação, que são as caracteres.
Por último, algumas estratégias foram executadas para analisar a eficiência da rede neural desenvolvida e seu progresso no decorrer do treinamento.

## Experimentos
Após testes de diversos valores para o número de camadas de camadas escondidas e de neurônios em cada uma delas, os valores selecionados foram os que obtiveram o melhor resultado, como será discutido posteriormente. Para essas escondidas, a função de ativação utilizada foi a ReLU, comumente escolhida para aplicações de visão computacional e a Softmax para a camada de saída da rede, a qual é uma função que recebe um vetor de número reais e retorna a distribuição probabilística proporcional ao exponencial dos valores de entrada.

Na execução do método fit para treinar a rede, foi-se estipulado que o número de épocas do treinamento seria interessante de 300. Ou seja, todos os dados seriam analisados, em suas totalidade, 300 vezes. Portanto, esse parâmetro influencia diretamente no resultado da rede. O seguinte é o de batch_size, que é quão grande deve ser a amostragem para cada alteração dos pesos dos neurônios, sendo, assim, de extrema importância na obtenção de bons resultados.

Como o dataset pode conter ruídos, indica-se valores para o batch_size não tão pequenos, justamente para evitar uma influência drástisca dessas coletas ruidosas. Além disso, o processo de treinamento pode se tornar muito longo, dependendo de quão baixo é este parâmetro.

<center>
<img src="https://i.postimg.cc/qMXK0N19/graf1.png"/>
<h5>Figura 4: Variação da perda de acurácia durante o treinamento da rede.
Fonte: Autor </h5></center>

É notável a tendência à estabilidade dessa perda após 50 épocas.

```py
<class 'list'>
Test score: 3.1143150519463423
Test accuracy: 0.6633825944170771```
```
A avaliação do modelo foi de, aproximadamente,  0,6633; este número é consideravelmente bom, se analisarmos que essa é uma primeira abordagem com a rede, podendo, então, ser otimizada com análises futuras.

<center>
<img src="https://static.vecteezy.com/system/resources/previews/000/184/397/original/letter-a-typography-background-vector.jpg"/>
<h5>Figura 5: Teste da rede utilizando uma imagem com o caractere A.
Fonte: Vecteezy </h5></center>
A rede foi capaz de acertar que a letra da Figura 5 é "A". Por fim, estes foram os testes realizados para averiguar a confiabilidade do treinamento.

```py
Predicted letter: [0]
```
## Referência
Schmidhuber. Deep Learning in Neural Networks: An Overview. Neural Networks, Volume 61, January 2015, Pages 85-117 (DOI: 10.1016/j.neunet.2014.09.003), published online in 2014.
