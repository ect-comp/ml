# Guia do Perceptron

O objetetivo do seguinte guia é a implementação de uma classe na linguagem de programação python que implemente o algoritmo de aprendizado supervisionado **Perceptron**.

São necessários alguns conhecimentos prévio do paradigma de orientação à objetos em Python.  

## Dica 

Utilize a biblioteca de programação *numpy* para instanciar eventuais arrays e matrizes, pois a mesma implementa métodos muito úteis que facilitam e aceleram o desenvolvimento.

## Criação da Classe Perceptron

### Etapa 1

Crie a classe Perceptron e em seu construtor receba e inicie as seguintes argumentos: tamanho da entrada, quantidade de épocas para o treinamento e taxa de aprendizagem. 

Dentro do construtor inicie um vetor para armezar os pesos do percepton, esse vetor deve possuir *tamanho da entrada* mais uma possições (esse acréscimo é necessário para que o *bias* seja tratado operado juntamente com os pesos, sendo assim o *bias* será o primeito elemento do array) todas inicialmete iguais a zero.

```python
class Perceptron(object):

    def __init__(self, input_size, epochs, learning_rate):
        # Seu código para inicar os argumentos 
```

### Etapa 2

Na classe Perceptron implemente um método para a função de ativação do Perceptron, esse método deve retornar 1 se a entrada for maior ou igual a zero ou 0 caso contrário.

```python
    def ativacao(self, entrada):
        # Seu código
```

### Etapa 3 

Na classe Perceptron implemente um método para a predição de uma saída dado uma entrada, para isso o método deve fazer a soma do bias com produto escalar da entrada com o vetor de pesos (sem o bias) e retornar o resultado da aplicação da função de ativação nessa soma.

```python
    def predizer(self, entrada):
        # Seu código
```

### Etapa 4

Por fim deve-se implementar um método para o treinamento do Perceptron, tal método deve receber como parâmetros as entradas a serem ajustadas e seus rótulos.

O método deve executar as seguintes intruções *quantidade de épocas* vezes para cada par de entrada, rótulo:

* Obtenha uma predição inicial para a entrada
* Atualize o vetor vetor de pesos de acordo com a equação w = w + taxa_de_aprendizado * (rótulo  —  predição) * entrada
* Atualize o *bias* de acordo com a equação bias = bias + taxa_de_aprendizado * (rótulo  —  predição)

```python
def train(self, entrada, rotulo):
        for _ in range(self.epocas):
            for entrada, rotulo in zip(entrada, rotulo):
                # Seu código
```

## Testes do Perceptron

A seguir segue a descrição do processo de uso do perceptron para realizar uma classificação binária com duas entradas, os dados de treino e o resultado esperado estão na imagem a seguir.

![alt Porta AND](https://raw.githubusercontent.com/ect-info/ml/tree/master/guias/Perceptron/and.jpg)

```python
# Cria o array de entradas
entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Cria um rotulo para cada entrada
rotulos = np.array([0, 0, 0, 1])

# Instancia um Perceptron
perceptron = Perceptron(tamanho_entrada = 2, epocas = 100, taxa_aprendizado = 0.01)

# Treina o Perceptron para os dados de entrada
perceptron.train(entradas, rotulos)


testes = np.array([
    [0.5, 0.5],
    [0.1, 0.3],
    [1.5, 1.5],
    [2.4, 2.5]
])

for i in testes:
    print("Predicao para {} foi {}".format(
            i, perceptron.predict(i)
        )
    )
```

## Caso fique perdido 

No caso de se perder durante a implementação, aqui esta um [exemplo](https://github.com/ect-info/ml/tree/master/guias/Perceptron/Perceptron.py) do perceptron implementado no [dataset Iris](https://archive.ics.uci.edu/ml/datasets/iris).






