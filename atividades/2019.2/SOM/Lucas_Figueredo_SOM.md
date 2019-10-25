# Identificação dos perfis de alunos na disciplina Lógica de Programação utilizando Self Organizing Maps

## Introdução
<p align="justify">
A equipe de desenvolvimento é formada por Lucas Figueredo Varela Alves , discente da disciplina de Tópicos Avançados em Informática I, ministrada pelo docente Orivaldo Vieira de Santana. 
</p>
<p align="justify">
O blog EdSurge Independent publicou, em 2018, um artigo intitulado "The Central Problem in Higher Education", o qual trata das dificuldades enfrentadas por integrantes do ensino superior. Dentre os tópicos levantados, pode-se citar a fraca coesão entre os componentes curriculares postos à disposição ou designados para os discentes. Essa baixa aplicabilidade dos conceitos absorvidos resulta em uma problemática no que tange a maior função social da universidade: formar cidadãos. 
Ademais, trazendo o contexto do problema para o Brasil, faz-se evidente que a educação de base do país possui déficits. Dentre eles, o disnível existente entre os ingressantes de universidade advindos de escolas particulares e os de escola pública proporciona um cenário de altíssima complexidade, no que diz respeito a dar continuidade ao progresso acadêmico do discente.
Nesse prisma, a capacidade de analisar o perfil dos estudantes e, por tanto, perceber, o mais cedo possível, que ele está enfrentando alguma dificuldade na disciplina em questão pode ser uma peça chave no avanço da busca por soluções para esse problema.

<p align="justify">
A sugestão é, então, uma tecnologia treinada a partir do Self Organizing Maps (SOM) à guisa de identificar padrões de comportamento e desempenho de alunos na disciplina de Lógica de Programação (LoP).
</p>

## Metodologia 
<p align="justify">
No campo do aprendizado de máquina, o Self-Organizing Map desenvolve uma rede neural artificial, a partir de uma aprendizagem não supervisionada, capaz de produzir uma redução dimensional, normalmente para duas, dos dados de entrada. 
Um dos maiores diferenciais desse tipo de rede é a preservação dos caracterísitcas topológicas dos dados, visto que a distribuição dos neurônios é composta por regiões de transição entre picos e vales, como pode ser visto na Figura 1. Essa abordagem se torna ainda mais interessante se o intuito for melhorar a visualização dos dados de alta dimensão. 
Por conseguinte, fez-se jus utilizar essa rede para analisar o perfil dos alunos em LoP. Em termos de treinamento da rede, ele funciona a partir de aprendizado competitivo e a distância Euclidiana é utilizada para computar a proximidade de um novo dado ou entre os dados de treinamento. 
</p>
<center>
<img src="https://www.viscovery.net/bilder/somine/UnfoldedMap.png"/>
<h5>Figura 1: Visualização das propriedades topológicas das entradas na rede SOM.
Fonte: viscovery.net </h5></center>

</p>
<p align="justify">
O banco de dados utilizado contém dados do número de questões submetidas ao sistema de avaliação da disciplina por semana. A ideia, por tanto, é buscar uma combinação de semanas que melhor represente e distingue os alunos com menor desempenhos dos demais. Vale salientar que é de extrema relevância que essa identificação seja feita o mais cedo possível, por tanto, buscam-se semanas do início/meio do semestre letivo.
</p>

## Códigos 
<p align="justify">
A etapa inicial é a de pré-processamento, na qual se faz necessário alterar as variáveis categóricas de respostas, as quais correspondem à situação do aluno em diversos casos. Para o nosso estudo, entende-se que só é interessante a abordagem da aprovação ou não do discente. Por tanto, modifica-se o database inicial:
</p>

```py
# Importing the dataset
dataset = pd.read_csv("https://raw.githubusercontent.com/ect-info/ml/master/dados/lop_submissao_semana.csv")
dataset = dataset.replace("APROVADO", 1)
dataset = dataset.replace("APROVADO POR NOTA", 1)
dataset = dataset.replace("REPROVADO POR NOTA", 0)
dataset = dataset.replace("REPROVADO POR MÉDIA E POR FALTAS", 0)
dataset = dataset.replace("REPROVADO", 0)
```
<p align="justify">
Em seguida, foi-se definido que as 8 primeiras semanas seriam analisadas. Apesar dos dados terem uma escala coerente, preferiu-se a aplicação de Feature Scaling para melhor visualização futura.
</p>

```py
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
X = sc.fit_transform(X)
```

<p align="justify">
Após isso, pode-se treinar a rede. Os parâmetros para a MiniSOM são: tamanho da rede na coordenada x e y, quantidade de parâmetros de entrada, taxa de aprendizado e o sigma, que é o raio de distância entre os neurônios vizinhos. Os pesos iniciais são introduzidos de forma aleatória aos neurônios e o número de iterações foi definido como 20000.
</p>

```py
# Training the SOM
redex = 6
redey = 6
from minisom import MiniSom
som = MiniSom(x = redex, y = redey, input_len = 8, sigma = 1.0, learning_rate = 0.45)
som.random_weights_init(X)
som.train_random(data = X, num_iteration =20000)
```
## Experimentos 
<p align="justify">
Com a rede treinada, o objetivo passa a ser a visualização dos dados. O primeiro gráfico mostra o mapa e a marcação de pontos dos neurônios vencedores (que melhor representa um determinado vetor) para cada valor de entrada.
</p>
<center>
<img src="https://i.ibb.co/4Tc39rv/map1.png"/>
<h5>Figura 2: Mapa com marcação dos pontos de entrada da rede.
 </h5></center>

<p align="justify">
A ideia posterior foi de encontrar quantos alunos foram aprovados em cada neurônio formado, o resultado foi:

```py
Total de alunos por neurônio:
[[193.  37.  38.  45.  18.  23.]
 [ 10.  25.  27.  22.  15.  17.]
 [ 69.   6.  21.  14.   8.  13.]
 [ 10.  49.  18.  19.  12.  11.]
 [ 16.  33.  16.  11.  23.  20.]
 [ 20.  14.  24.  27.  10.  14.]]
Aprovados por neurônio
[[60. 22. 17. 29. 17. 22.]
 [ 6. 14. 18. 20. 14. 17.]
 [49.  3. 16. 11.  8. 13.]
 [ 7. 31. 12. 17. 11. 10.]
 [13. 25. 12. 10. 20. 18.]
 [19. 12. 21. 26.  8. 11.]]

```
</p>

<p align="justify">
Por último, os dois últimos últimos mapas se complementam e tornam capaz a visualização do número de questões submetidas por semana em cada neurônio e a proporção de aprovados (em azul) para cada neurônio.
</p>
<center>
<img src="https://i.ibb.co/9TGY26d/map2.png"/>
<h5>Figura 3: Mapa com a proporção de aprovados por neurônio.
 </h5></center>

</p>
<center>
<img src="https://i.ibb.co/Mcm1D3d/map3.png"/>
<h5>Figura 4: Mapa com o número de submissões média dos alunos em cada semana por neurônio.
 </h5></center>

<p align="justify">
Em suma, pode-se verificar que a região noroeste do mapa é composta, principalmente, por alunos que não obtiveram sucesso na disciplina. O mapa da figura 4 evidencia, ainda, que estes não tiveram uma frequência de submissões positiva em comparação com a de outros clusters da rede.
Outra observação interessante é que dois neurônios do canto direito são compostos apenas por aprovados e suas submissões nas semanas são mais homogêneas que as dos demais. Por tanto, constata-se o impacto da continuidade e assiduidade das submissões de questões durante o semestre letivo. 
</p>
