# Atividades Avaliativas 

As atividades devem ser entregues no github do aluno em um repositório com o nome "ect2702-ml". Um diretório deve ser criado para cada atividade com os códigos fontes que reselvem a atividade. O nome do diterório será especificado na descrição de cada atividade. As atividades podem ser feitas em grupo, mas cada estudante deve desenvolver seu próprio código. 

## Unidade I

Prazo máximo para entrega da atividades desta unidade: 19/09/2018. 

### Regressão Linear 
Estimar o sálario de um funcionário de uma determinada empresa baseado em seus anos de experiência. 

* Nome da pasta: rl-salario;
* Código de apoio: https://colab.research.google.com/drive/1J3-jnp7-en_OjvCG-He8UcD-Or8g_nxm.

### Perceptron 

Implementar um perceptron e testar com uma base de dados simples. Sugestão de problema para testar a implementação, *extraído das notas de aula da Professora Teresa Ludemir, CIN-UFPE*: 

* Dada uma rede do tipo Perceptron formada por um neurônio com três terminais de entrada, utilizando pesos iniciais w0 = 0.4, w1 = -0.6 e w2 = 0.6, limiar θ = 0.5 e uma taxa de aprendizado = 0.4. Considere que o limiar será sempre multiplicado por -1. responda os itens abaixo:
  * Ensinar a rede a gerar a saída -1 para o padrão 001 e a saída +1 para os padrão 110; 
  * A que classe pertencem os padrões 111, 000, 100 e 011?
* Manipulação básica de vetores: http://bit.ly/2w6xgfd;
* Roteiro para implementação: https://github.com/ect-info/ml/tree/master/guias/Perceptron, contribuição de Richardson (https://github.com/vanluwin);
* Nome da pasta: perceptron; 

### Classificação com MLP ou SVM 

Escolher um problema, gerar a base de dados e implementar um código para a classíficação destes dados. 

Nesta atividade será levado em conta: 
* A complexidade da base de dados desenvolvida ou obtida; 
* O esforço de implementação do classificador; 
* A taxa de acerto do classificador. 

#### Sugestões de atividades: 
* Sensor de Cores: 
  * Os dados devem ser obtidos de um sensor de cor de arduino e uma base de dados no formato csv deve ser gerada; 
  * O código de reconhecimento de cores deve estar embarcado no arduíno ou em um raspberry pi.
  
* Róbtica: 
  * Coletar dados de sensor utlrassom de um robô com rodas para o comportamento de desviar de obstáculos ou seguir obstáculos; 
  * Implementar a etapa de resposta da rede (fase *forward*) no Arduino.  
  
* Reconhecedor simplificado de gestos:
  * Coletar dados de gestos formando os números de 0 a 9; 
  * A base de dados deve conter pelo menos 20 gestos de cada número; 
  * O reconhecimento pode ser feito no computador ou em um raspberry pi. 

* Mineração de dados: 
  * Estimar a tendência de conlusão do curso de graduação; 
  * Considerando os dados do desempenho do aluno no primeiro semestre e a informação de conclusão ou desistência do curso gerar uma base de dados para classificação da tendência de finalização do curso; 
  * O desempenho pode ser medido pela média final dos componentes curriculares cursados; 
  * Essa base de dados é uma sugestão inicial. 
  
* Nome da pasta: classificacao-mlp ou classificacao-svm 



