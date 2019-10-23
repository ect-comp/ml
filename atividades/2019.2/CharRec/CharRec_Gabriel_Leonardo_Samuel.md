# Classificador de imagens usando Pytorch

## Introdução
O presente trabalho foi dividido em duas etapas: i) A primeira etapa se baseia em processamento de imagem. O usuário vai escrever em um papel letras maiúsculas ou minúsculas e números. Uma API de pré-processamento será responsável por: Corrigir a iluminação, fazer transformações morfológicas para aumentar o tamanho das bordas das letras e números, detectar bordas e por fim realizar um segmentação nas letras e números e separa-las em novas imagens.; ii) A segunda etapa se baseia em treinar um modelo de deep-learning para identificar as letras e números. Esse modelo de DeepLearnig foi construido utilizando a biblioteca Pytorch, e os dados de treinamento foram obtidos no dataset http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ , T. E. de Campos, B. R. Babu and M. Varma. Character recognition in natural images. In Proceedings of the International Conference on Computer Vision Theory and Applications (VISAPP), Lisbon, Portugal, February 2009. As imagens obitdas na etapa "i" foram submetidas a testes para o modelo treinado.

Métodos de Aprendizado Profundo (Deep Learning) são atualmente o estado-da-arte em muitos problemas possíveis de se resolver via aprendizado de maquina, em particular
problemas de classificação. No entanto, ainda há pouco entendimento de como esses métodos funcionam, porque funcionam e quais as limitações envolvidas ao utilizá-los.técnicas de Aprendizado Profundo tem revolucionado diversas áreas de aprendizado de máquina, em especial a visão computacional. Isso ocorreu principalmente por dois motivos: a disponibilidade de bases de dados com milhões de imagens [8, 46],
e de computadores capazes de reduzir o tempo necessário para realizar o processamento dessas bases de dados.Os pré-requisitos necessários para entender como Deep Learning funciona, incluem conhecimentos básicos de Aprendizado de Máquinas (ML) e Processamento de Imagens (IP), em particular conhecimentos básicos sobre aprendizado supervisionado, classificação, redes neurais Multilayer Perceptron (MLP), aprendizado não-supervisionado, fundamentos de processamento de imagens, representação de imagens, filtragem e convolução. HARA, Carmem S.; PORTO, Fábio; OGASAWARA, Eduardo. TÓPICOS EM GERENCIAMENTO DE DADOS E INFORMAÇÕES 2015.


### Contribuidores

* Samuel Amico Fidelis, aluno da Universidade Federal do Rio Grande do Norte do curso de bacharelado em engenharia mecatrônica. Para eventuais dúvidas, entrar em contato pelos
meios abaixo:
- [Github](https://github.com/samuelamico/MachineLearning)
- [Site](https://samuelamico.github.io/)

* Leonardo Queiroz, aluno de bacharelado em engenharia mecânica na Universidade Federal do Rio Grande do Norte, contato em:
- [Github](https://github.com/leocqueiroz)

* Gabriel Varela, aluno da Universidade Federal do Rio Grande do Norte do curso de bacharelado em engenharia mecatrônica. Contato em:
- [Github](https://github.com/gabrielvrl)
- [Site](https://gabrielvrl.github.io/)

### Metodologia
A metodologia será divida em duas partes, a aquisição e pré-processamento de imagens do usuário, e o modelo de DeepLearning utilizado:

a) Após o usuário escrever na folha ele passa para o computador a imagem da folha e então o código CropDigit.py irá ser executado, onde a imagem irá passar pelos seguintes processos:

![Processamento](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/PreProcessamento.png)

b) A segunda parte é baseada em utilizar o modelo pré-treinado da ResNet50 para treinar o nosso identificador de letras e números. O modelo da ResNet foi baixado com ajuda do próprio Pytorch e foi mudado apenas
a FullyConnected Layer, onde a nossa saída se baseia em 62 tipos de letras (entre elas  maiúsculas ou minúsculas) e os 10 números. Foi utilizado um pré-processamento nas imagens de treinamento, esse pré-processamento se baseia em: padronizar o tamanho da imagens e transforma-las em Tensores. O treinamento pode ser feito em CPU ou GPU dependendo do usuário. A imagem abaixo mostra o modelo ResNet utilizado:

![ResNet](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/ResNet.PNG)

![ResNetFC](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/ResNetFC.PNG)

Para o treinamento, foi utilizado um batchsize de tamanho igual a 12, utilizado apenas 10 épocas de treinamento. Foi obitdo as seguintes métricas para o treinamento:

![Treinamento1](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/Treinamento.PNG)


![Treinamento2](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/Treinamento2.PNG)

Como pode ser visto, o treinamento ainda não foi suficiente e também a quantidade de imagens no dataset não foi suficiente para se obter uma boa acurácia.


### Experimentos e Códigos

Os códigos para as duas etapas do Problema estão contidos no GitHub de Samuel: [Github](https://github.com/samuelamico/MachineLearning/tree/master/DeepLearnin%20-%20HandWritten).


#### Etapa 1

* Código e Experimento para a primeira etapa: Foi criada uma classe para realizar todo o pré-processamento necessário para melhorar a detecção de borda e assim conseguir obter o máximo de letras e números na folha de papel do usuário. As imagens abaixo ilustram as etapas do experimento:

![letrasOriginal](https://github.com/samuelamico/MachineLearning/blob/master/DeepLearnin%20-%20HandWritten/Images/letras.png)

A imagem acima é a folha de papel onde o usuário escreveu, a partir dai, a classe pré-processamento entrara em ação:

```py
class Preprocess():
    def __init__(self,image,gamma=0.3):
        self.image = image
        self.gamma = gamma
    
    def adjust_gamma(self):
        invGamma = 1.0 / self.gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(self.image, table)

    def sharpening(self,image):
        smoothed = cv2.GaussianBlur(image,(3,3),5)
        unsh = cv2.addWeighted(image,1.5,smoothed,-0.5,0)
        return unsh


    def resizer(self,image):
        img_re = image
        scale_percent = 160 # percent of original size
        width = int(img_re.shape[1] * scale_percent / 100)
        height = int(img_re.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(img_re, dim, interpolation = cv2.INTER_CUBIC)
        return resized

    def Filter(self):
        image_gamma = self.adjust_gamma()
        image_sharp = self.sharpening(image_gamma)
        resized = self.resizer(image_sharp)
        return resized

```

e a função de retirar sombra:

```py
## Retirar sombra
rgb_planes = cv2.split(img)
result_planes = []
result_norm_planes = []
dst = np.zeros(shape=(5,2))
for plane in rgb_planes:
    dilated_img = cv2.dilate(plane, np.ones((13,13), np.uint8))
    bg_img = cv2.medianBlur(dilated_img,21)
    diff_img = 255 - cv2.absdiff(plane, bg_img)
    norm_img = cv2.normalize(diff_img, dst ,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8UC1)
    result_planes.append(diff_img)
    result_norm_planes.append(norm_img)
```

resulta na imagem abaixo:

![letrasSemSombra](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/NoShaw.png)

e após a correção gama, a imagem resultante:

![letrasGama](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/gamma_corection.png)

Será então aplicado a detecção de borda e a segmentação em cada nova imagem que contenha as letras ou números detectados:

```py
# Find the Edges:
res = cv2.findContours(img_erosion.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
if cv2.__version__.split(".")[0] == '3':
    im2, contours, hierarchy = res
else:
    contours, hierarchy = res

rects = []
for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    x, y, w, h = cv2.boundingRect(approx)
    if (h >= 16):
        # if height is enough
        # create rectangle for bounding
        rect = [x, y, w, h]
        rects.append(rect)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 1);

cv2.imwrite('letras_final.png', img)
print(rects)

NI = 0
path = 'ResultLetter/'
for posi in rects:
    x,y,w,h = (posi[0],posi[1],posi[2],posi[3])
    #print(x,y,w,h)
    letter_img = img_erosion[y:y+h,x:x+w]
    # Interpolation:
    preproce = Preprocess(img_erosion)
    letter_img = preproce.resizer(letter_img)
    cv2.imwrite(path+'2Letter'+str(NI)+'.png',letter_img)
    NI+=1
```
As imagens abaixo ilustram como ficou a segmentação e como ficou as imagens das letras separadas: obs: observe que a letra S não foi detectada.


![letrasFinal](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/letras_final.png)

Exemplos de letras detectadas:

![letrasP](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/2Letter10.png)

![letrasE](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/2Letter22.png)

![letrasA](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/2Letter25.png)




#### Etapa 2

A primeira etapa antes de começar a treinar é dividir o dataset de treinamento e teste e aplicar as transformações:

```py
data_dir = 'Dataset/train'
def load_split_train_test(datadir, valid_size = .2):
    train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(),
                                       ])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=64)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    return trainloader, testloader
trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)
```

Logo em seguida o modelo ResNet50 é carregado para o programa e a mudança na FC é realizada, além da escolha do otimizado ADAM:

```py
for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 62),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)
```

O treinamento então é feito e o modelo é salvo para realizar os testes:

```py
epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'eminist.pth')
```

A validação final do experimento é então realizada com o modelo já salvo e treinado e com as imagens extraidas da folha de papel e mais algumas outras que foram obtidas para teste. As imagens de teste são salvadas em um novo diretório e escolhidas algumas para a validação:

```py
to_pil = transforms.ToPILImage()
images, labels,classes = get_random_images(5)
fig=plt.figure(figsize=(10,10))
for ii in range(len(images)):
    image = to_pil(images[ii])
    index = predict_image(image)
    sub = fig.add_subplot(1, len(images), ii+1)
    res = int(labels[ii]) == index
    sub.set_title(str(classes[index]) + ":" + str(res))
    plt.axis('off')
    plt.imshow(image)
plt.show()
```


![Validação](https://github.com/samuelamico/MachineLearning/blob/master/Graficos/Teste.PNG)

Portanto, a etapa 1 teve um bom resultado em extrair as letras ou números, o treinamento foi realizado usando ResNet50, porém o resultado não foi tão satisfatório devido a pouca quantidade de imagens para o dataset e as poucas épocas utilizadas no treinamento.
