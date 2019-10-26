# Previsão da situação final (aprovação ou não) de alunos na disciplina de Lógica de Programação (LoP) utilizando Self-Organizing Map (SOM)

## Introdução

Este trabalho foi realizado por Bruno Guedes Spinelli, aluno do curso de Engenharia Biomédica da UFRN.

Existe uma grande evasão de alunos durante a disciplina de Lógica de Programação (LoP) derivada de diversos motivos, mas acredita-se que o maior causador dessa evasão seja a desmotivação decorrente das notas baixas do aluno. O objetivo deste trabalho é criar rede neural do tipo Self-Organizing Map (SOM) para tentar prever a situação final do aluno (reprovado ou aprovado) na disciplina LoP a partir da quantidade de submissões semanais do aluno, assim tornando possível a realização de feedbacks durante a disciplina que estimulem uma maior interação do aluno com o conteúdo ministrado e consequentemente aumentando sua chance de tirar notas melhores e diminuindo desta forma a evasão dos alunos.

A base de dados utilizada para treinar e testar a rede pode ser encontrada clicando [aqui], nela podemos observar as seguintes classificações de dados:

* **anoSubmissao, anoSigaa e anoTurma:** período em que foram realizadas as submissóes de atividades.
* **semana 1 - 21:** quantidade de atividades submetidas por semana.
* **descricaoTurma:** descrição da disciplina, turma e subturma.
* **turma:** descrição da turma do aluno.
* **situacao:** situação final do aluno.

## Metodologia

O Self-Organizing Map (SOM) é um tipo de rede neural artificial não supervisionada, bastante utilizada para a visualização de dados de alta dimensionalidade, treinada para produzir uma representação capaz de descrever as amostras de entradas em baixa-dimensão (geralmente duas dimensões) chama de mapa. A principal diferença encontrada no SOM em comparação com outros tipos de rede neural artificial é quanto a sua forma de aprendizagem, na rede SOM a aplica-se o aprendizado competitivo, em que os pesos dos neurônios são comparados com os pesos do dado de entrada e o vencedor (aquele que possuir os pesos mais próximos dos do dado de entrada) tem seus pesos e  os pesos de seus vizinhos atualizado.

![som1]

**Figura 1 - Representação de um SOM**

Para tentar prever a situação final do aluno através da quantidade de submissões de atividades por semana, o treino da rede SOM presente neste trabalho foi realizada com as seguintes colunas da base de dados fornecida:

**semana 1 - 21** 

## Códigos

A rede neural foi codificada utilizando a linguagem de programação Python. O programa utilizou as bibliotecas ```pandas```  para auxiliar no processo de importar o arquivo de dados , a biblioteca ```numpy``` para realizar operações matemáticas e também foi utilizado a biblioteca ```minisom``` para criar a rede:

```py
import pandas as pd
import numpy as np
from minisom import MiniSom
dataset =  pd.read_csv("https://raw.githubusercontent.com/ect-info/ml/master/dados/lop_submissao_semana.csv",index_col=False )
dataset.head()
```
Os dados da coluna ¨situacao¨ foram substituídos por um classificador binário, sendo dado o valor 1 para alunos aprovados e o  valor 0  para alunos reprovados, e em seguida os classificadores para a realização do treino e teste da rede foram selecionados, para X (submissão de atividades por semana) foram selecionados os dados das colunas ¨semana 1¨  até ¨semana 21¨ (colunas de 2 a 22) e para Y (situação final do aluno) foi selecionados os dados da coluna ¨situacao¨ (coluna 25):

```py
dataset.replace('APROVADO',1,inplace=True)
dataset.replace('APROVADO POR NOTA',1,inplace=True)
dataset.replace('REPROVADO',0,inplace=True)
dataset.replace('REPROVADO POR NOTA',0,inplace=True)
dataset.replace('REPROVADO POR MÉDIA E POR FALTAS',0,inplace=True)

X = dataset.iloc[:, 2:23].values
Y = dataset.iloc[:, -3].values
```

Logo após, foi realizada a criação da rede SOM em que foram passados parâmetros como tamanho da rede (10 x 10), quantidade de parâmetros de entrada (21), sigma (1.0) e a taxa de aprendizado (0,5) e em seguida o treinamento da rede:

```py
som = MiniSom(x = 10, y = 10, input_len = 21, sigma = 1.0, learning_rate = 0.5)
som.random_weights_init(X)
som.train_random(data = X, num_iteration = 40000)
```

## Experimentos 

Após o treino da rede, foram geradas duas matrizes, uma contendo o total de alunos enquadrados pelos pesos dos neurônios e o outro o total de alunos aprovados enquadrados nos pesos dos neurônios:

![resultado1]

**Figura 2 - Alunos totais por neurônio da rede x alunos aprovados por neurônio da rede**

Em seguida foram gerados gráficos de pizza representando a quantidade de alunos aprovados (azul) x alunos reprovados (amarelo) por neurônio:

![resultado2]

**Figura 3 - Alunos aprovados x reprovados por neurônio**

Também foi selecionado um neurônio que demonstra pesos que levam a alta chance de aprovação do aluno na disciplina de LoP:

![resultado3]

**Figura 4 - Alta chance de aprovação na disciplina de LoP: quantidade de submissões por semana**

Por último, alta chance de reprovação do aluno na disciplina de LoP:

![resultado4]

**Figura 5 - Alta chance de reprovação na disciplina de LoP: quantidade de submissões por semana**

<!-- Links -->

[aqui]: https://github.com/ect-info/ml/blob/master/dados/lop_submissao_semana.csv
[resultado1]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/2019-10-25-222310_1920x1080_scrot.png
[resultado2]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/2019-10-25-223649_1920x1080_scrot.png
[resultado3]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/2019-10-25-224310_1920x1080_scrot.png
[resultado4]: https://github.com/brunospinelli/ECT-TAI1/blob/master/imagens/2019-10-25-224311_1920x1080_scrot.png
[som1]: http://ivape3.blogs.uv.es/files/2015/03/kohonen1.png
