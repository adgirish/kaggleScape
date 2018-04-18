
# coding: utf-8

# **This notebook is for a Portuguese speaking audiance as part of a training session. Soon I will post it in English. **
# 
# >"Então você não se lembra de um mundo sem robôs.  Houve um tempo quando a humanidade enfrentou o universo sozinha e sem amigo. Agora ela tem criaturas para ajudá-la; criaturas mais fortes que si mesma, mais confiáveis,  mais úteis e absolutamente devotas. A humanidade não mais está sozinha. Você já pensou nisso desta forma?" <br>
# I, Robot - Issac Asimov, 1950
# 
# 
# # Uma breve história dos algoritmos que aprendem
# 
# <br><br>
# **Bem-vindos ao Laboratório Introdutório de Machine Learning!**
# <br><br>
# 
# Esse é um dos livros que vamos  usar como referência [Python Machine Learning - Second Edition,
# Raschka & Mirjalili, September-2017](https://www.packtpub.com/big-data-and-business-intelligence/python-machine-learning-second-edition)
# 
# 
# O primeiro passo para iniciar nossos estudos  é compreender que **Machine Learning (ML)** é um sub-campo de pesquisa da **Inteleligência Artificial (IA)** e,  portanto, não é necessariamente seu sinônimo como erroneamente sugerem alguns desavisados por ai.  ** Deep Learning (DL)** é um dos tópicos de **Redes Neurais (NN's)** que por sua vez são uma das sub-áreas de **ML**.  Não cometa o erro de confundir indistintamente Deep Learning com Machine Learning.
# ![](https://blogs.nvidia.com/wp-content/uploads/2016/07/Deep_Learning_Icons_R5_PNG.jpg.png)
# 
# 
# 
# # Peceptron
# Algoritmos de aprendizagem não são um tema novo.  A definição de neurônio artificial, o **perceptron**, foi estabelecida no final da década de 50 (The Perceptron: A Perceiving and Recognizing Automaton, F. Rosenblatt, Cornell Aeronautical Laboratory, 1957) e pode ser resumida na função abaixo:
# 
# ![](https://www.dropbox.com/s/s0uvoloszvkg83x/00-Perceptron.jpg?dl=1)
# 
# De forma simplificada, a saída de um neurônio artificial é igual a soma do produto das entradas ***x*** pelos pesos ***w*** aplicados a cada entrada.   As entradas ***x*** de um neurônio artificial equivalem aos dentritos de um neurônio biológico e a soma **$\sum_{j=0}^m x_{j}w_{j}$** é o estímulo resultante no axônio, definido por um limiar interno do neurônio (**threshold**) que vai determinar a sua "sensibilidade" ou quando será ativado ou não. Em ML preferencialmente utilizaremos a notação matricial  **$w^Tx$** onde o produto é dado pela transposta de *w* por *x*. A utilização de matrizes permite maior eficiência computacional e simplificação dos códigos de ML. 
# 
# <br><BR>
# Aqui temos a representação gráfica do perceptron:
# 
# ![](https://www.dropbox.com/s/yxvrkm7kk1r991e/01-Perceptron.jpg?dl=1)

# ![](https://www.sololearn.com/avatars/b2b6905b-4e53-412a-bcb8-22bfef2bcec5.jpg)
# 
# # Quando as máquinas aprendem...
# 
# A aprendizagem se traduz em encontrar pesos que aplicados aos valores de entrada resultem em um determinado valor de saída esperado.  Ainda analisando o gráfico do perceptron acima, vale notar que por questões de convenção e cálculo a entrada **$x_{0} $** é fixada com o valor ***1*** e seu o peso **$w_{0} $**  é chamado de **bias**.   Em uma rede neural de apenas uma entrada teríamos a seguinte equação equivalente z =  $w_{0}$ + $w_{1}x_{1}$.  Se voltarmos às aulas de matemática fundamental veremos que essa é exatamente uma **equação reduzida da reta **, onde $w_{0}$ define a "altura" da reta e  $w_{1}$ define sua inclinação no gráfico.   
# 
# ![](https://www.dropbox.com/s/cdai4n28jp5m5wp/simple_regression.png?dl=1)
# 
# 
# O que os algoritmos de ML fazem é buscar de forma automática a equação que melhor representa o conjunto verdade (**y**) para um conjunto de observações ou amostras de entradas.  Uma forma de encontrar a melhor equação é através do cálculo sucessivo da diferença entre os valores gerados pela equação "aprendida" (**ŷ**) e os valores reais observados (**y**). Essa diferença chamamos de **Erro** ou **Perda**. As funções de perda  ou **loss functions** são um importante elemento na construção de algoritmos inteligentes. Em outras palavras, podemos afirmar que a aprendizagem de máquina é essencialmente uma tarefa de otimização de funções.  Atualmente os principais frameworks de ML implementam diversos algoritmos de otimização, sendo o Gradiente Descendente Estocástico ([SGD](http://ruder.io/optimizing-gradient-descent)) um dos mais populares.
# 
# O processo de ajustar pesos através de algoritmos de otimização de função é denominado **fit (treino)**, e cada rodada de ajustes é chamada de **Epoch (Época)**. O ajuste geralmente é feito usando um determinado número de amostras por vez que chamamos de **Batch (Lote)** .
# 
# No gráfico acima temos um problema onde os valores de solução podem ser linearmente correlacionadas com as amostras de entrada. Neste caso um algoritmo de ** Regressão ** poderia ser aplicado, mas existem varios tipos de algoritmos de ML e cada um vai funcionar melhor em determinados cenários.  Daí o teorema  **No Free Lunch**  em Machine Learning de David H. Wolpert, que nos recorda de que nenhum algoritmo de ML é universalmente melhor que todos os outros em todos cenários (The Lack of A Priori Distinctions Between Learning Algorithms, Wolpert and David H, 1996).
# 
# 
# 
# # Rolling in the Deep
# 
# As redes profundas conhecidas como Deep Learning(DL) ganharam maior destaque a partir de 2012 com a vitória de um time universitário canadense em uma competição de classificação de Imagens, a [ImageNet](http://www.image-net.org/).
# 
# Mas a vitória deste time canadense está intimamente relacionada com avanços da década de 80, sendo um de seus expoentes o cientista de computação e psicologia cognitiva **Geoffrey Hinton** da Universidade de Toronto. Hinton é conhecido por temas como **Propagação Reversa (Backpropagation)**, **Máquina de Boltzman (Boltzmann Machine** e **Deep Learning**. 
# 
# Embora o termo Deep Learning já havia sido aplicado a redes neurais artificiais por Igor Aizenberg em 2000,  foi uma plublicação de Geoffrey Hinton e Ruslan Salakhutdinov em 2006 que chamou mais atenção ao mostrar como redes neurais poderiam ser pré-treinadas uma camada por vez, e então fazer ajustes finos por meio de Backpropagation.  Esse avanço contribuiu fortemente para a viabilidade das redes DL como hoje conhecemos.
# 
# Em 2012, Hinton e seus dois alunos Alex Krizhevsky e Ilya Sutskever entraram na competição ImageNet e ao fazerem uso de redes densas convolucionais (CNN's) e técnicas avançadas para reduzir overfitting (ajuste excessivo aos dados de treino que resulta em baixa generalização) conseguiram atingir um incrível patamar de erro de 16% contra os 25% alcançados até então com algoritmos classificadores existentes. Hinton e seus alunos criaram uma empresa que seria adquirida posteriormente pelo Google.
# 
# Abaixo o gráfico da arquitetura de sua rede **[AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)** . Esta rede foi treinada por cerca de 5 a 6 dias usando um dataset de milhões de imagens classificadas milhares de classes. A equipe da AlexNet além da arquitetura inovadora utilizou duas placas de video GTX 580 (GPU) para poder suportar a alta demanda de processamento desse tipo de rede. O poder de manipulação de matrizes de uma GPU é muito bemvindo com algoritmos de ML, já que no final das contas toda informação e aprendizagem resultam em matrizes de dados e pesos.
# 
# 
# ![](https://image.slidesharecdn.com/dlcvd2l4imagenet-160802094728/95/deep-learning-for-computer-vision-imagenet-challenge-upc-2016-7-638.jpg?cb=1470131387)
# 
# Nos anos seguintes empresas como Nvidia, Google, Microsoft, Baidu, Amazon, IBM, Ubber, Facebook e Tesla  entrariam de forma ainda mais agressiva na corrida tecnológica por plataformas de inteligência artificial mudando o nível do jogo para uma aposta de trilhões de dólares,  e criando com o apoio das diversas comunidades de código aberto os frameworks poderosos que estão hoje ao alcance de alguns cliques. Abaixo algum dos principais frameworks da atualidade:
# 
# ![](https://www.dropbox.com/s/lv9ooa3ur8pxc33/deep-learning-developer-frameworks-407.png?dl=1)

# # O que é Machine Learning?
# <br>
# A seguir vamos começar a entender um pouco mais sobre como funciona os principais tipos de algoritmos de ML,  quais são as estratégias de treino e etapas para construção destes algorítimos.
# 
# Então, uma pergunta importante a fazer é: o que é um algorítimo de aprendizagem de máquina? Uma boa definição, emprestada de [*Deep Learning*](http://www.deeplearningbook.org) (Goodfellow-et-al-2016),  seria ***"um algorítmo de aprendizagem de máquina é um algorítmo capaz de aprender com os dados"* **.
# 
# Ok, mas o que significa aprender? Tom Mitchell em seu livro* Machine Learning* (McGraw-Hill, New York. 97) nos ajuda com uma definição bem sucinta: *** “Um programa de computador é dito aprender de uma experiência E em respeito a alguma classe de tarefa T e medida de performance P, se sua performance em tarefas T, como medido por P, melhora com a experiência E".*** 
# 
# Todavia não podemos esquecer que Machine Learning é um campo em construção e muitos dos conceitos que hoje consideramos  verdade serão descartados nos próximos anos. O próprio [Geoffrey Hinton em entrevista com Andrew NG](https://www.youtube.com/watch?v=-eyhCTvrEtE) (outro nome bastante conhecido da galera de ML) diz *"Meu conselho é que leia alguma literatura (*de ML*) mas não leia demais... alguns dizem que você deveria passar vários anos lendo a literatura e só então começar a trabalhar em suas próprias idéias e isso pode ser verdade para alguns pesquisadores, mas para pesquisadores criativos eu penso que o que você quer é estudar uma parte da literatura e, então, notar o que todos estão fazendo errado... aquilo que você sente que não está correto, e ao contrário imaginar um jeito de fazer certo... e quando os outros disserem que não serve, apenas continue... tenho um bom princípio para ajudar as pessoas a continuarem que é: ou suas intuições são boas ou não, se são boas você deveria seguir-las e ao final terá sucesso, se não são boas não importa o que você faça... você deveria confiar nas suas intuições não há razão para não fazê-lo..."* (tradução livre)
# 
# 
# Portanto, a seguir veremos três grandes grupos de algoritimos de ML, mas utilize essa divisão apenas como ferramenta de compreensão já que alguns algoritmos atuais extravasam essas classificações.
# 
# 
# # Os três tipos de Machine Learning
# 
# Os algoritmos de Machine Learning podem ser agrupados em três tipos principais:
# 
# 
# ![](https://www.dropbox.com/s/btluyzv2e08djan/02-MLTipos.jpg?dl=1)
# 
# ## Supervised Learning
# O principal objetivo na **aprendizagem supervisionada** é "aprender" um modelo com base nos dados de treino rotulados que seja capaz fazer predições a respeito de dados novos ou de dados futuros. 
# 
# Quando os valores esperados são discretos, como por exemplo um algoritmo capaz de reconhecer se uma imagem é de um gato ou cachorro, dizemos que se trata de uma **Tarefa de Classificação** ou seja buscamos um modelo classificador.  Classificação é uma subcategoria da aprendizagem supervisionada na qual o foco é prever rótulos categóricos de novas instâncias baseado nas observações do passado.
# 
# A predição de valores contínuos, como por exemplo o preço de venda de um imóvel, é tratada por outra subcategoria de aprendizagem supervisionada a **Regressão**.  
# 
# 
# ## Reinforcement Learning
# Outro tipo de aprendizagem de máquina é o aprendizado por reforço. Em **reinforcement learning** o objetivo é desenvolver um **agente** que melhora sua performance baseado em sucessivas interações com o ambiente.  Diferentemente das funções de perda (loss functions) das técnicas supervisioanadas, aqui o feedback é dado por um sistema de recompensas que pune ou premia certos resultados (**reward function**) com base em certos estados do ambiente.
# 
# Um exemplo popular desta arquitetura de aprendizagem é uma Engine de Xadrez. Nela o agente decide uma série de movimentos de acordo com o estado do tabuleiro, a recompensa pode ser definida com base em diversos resultados como sobrepor uma peça inimiga ou tomar sua rainha, ou mesmo a vitória ou derrota final.
# 
# 
# 
# 
# ## Unsupervised Learning
# Na aprendizagem não supervisionada lidamos com dados não rótulados ou com informação cuja estrutura não é exatamente conhecida.   
# 
# Ao usarmos  técnicas de aprendizagem não supervisionada somos capazes de explorar a estrutura de nossas amostras e extrair informação significativa de como essas amostras se relacionam.  Uma das aplicações práticas deste tipo de aprendizagem é a segmentação (**clustering**) de clientes de acordo com suas preferências ou quaisquer outras características que tenhamos à disposição. 
# 
# Outro campo de aplicação da aprendizagem não supervisionada é redução de dimensionalidade de dados. A redução de dimensionalidade permite eliminar ruídos e comprimir informação resultando em economia de processamento e armazenamento de dados. 

# # Botando a mão na massa!
# 
# Agora que vimos em linhas gerais o que são algorítmos de ML, vamos começar com o primeiro passo no desenvolvimento de um sistema de ML:   A preparação e exploração de dados ou **exploratory data analysis (EDA)** termo também emprestado do campo de estatística.  
# 
# Nos exemplos vou usar um famoso dataset chamado Iris que possui 150 amostras de 3 tipos de flores e os tamanhos de suas pétalas.  Em ML essas características ou dados de entrada denominamos **features**.
# 
# *Você deve executar cada célula de código . Use Ctrl + Enter para executar e Shift + Enter para criar uma nova célula*
# 
# ## 1 - Bibliotecas 

# In[ ]:


#Usamos import para importar as bibliotecas e pacotes que vamos utilizar
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # library for draring charts

# a magic cell (%) abaixo permite exibir gráficos de forma interna adequadamente
get_ipython().run_line_magic('matplotlib', 'inline')

# Exibe a versão das biblioteca. Em alguns casos é importante em que versão está trabalhando
print("Numpy Version {}".format(np.__version__))
print("Pandas Version {}".format(pd.__version__))


# ## 2 - Caminho do Dataset

# In[ ]:


'''
O dataset que vamos usar foi adicionado automaticamente. Podemos adicionar qualquer dataset 
público do Kaggle com o botão "Add a Data Source" ou o seu próprio com "Upload a Dataset"

Os dataset adicionados serão postos no caminho "../input".
Abaixo executamos o comando linux ls através do python para listar os arquivos desta pasta:
'''
from subprocess import check_output
print('Arquivos Iris:')
print(check_output(["ls", "../input/iris"]).decode("utf8"))


# ## 3 - Carregando o Dataset

# In[ ]:


# Existem várias formas de se carregar um dataset para uso em ML as duas mais comuns:
# usar iblioteca numpy ou carregar um data frame do pandas como abaixo
df_iris = pd.read_csv('../input/iris/Iris.csv')


# In[ ]:


# Exibe as primeiras 5 linhas do dataframe
df_iris.head(5)


# ## 4 - Explorando os dados com gráficos do matplotlib
# 
# No dataset Iris temos na coluna Species os tipos das flores que vamos analisar. Para isso precisamos transformar as classes de flores em números, para podermos seguir com as análise

# **Exibindo a distribuição das classes**

# In[ ]:


# Verificamos os valores únicos para as espécies
print(df_iris['Species'].unique())

# Adicionamos uma nova coluna no data frame e mapeamos com um valor númerico por classe
# essa coluna é nosso target (a predição que nossa rede vai gerar)
df_iris['y'] = df_iris['Species'].map({'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica' : 3})

# Configuramos o gráfico
plt.xlabel('Sepal Length Cm')
plt.ylabel('Petal Length Cm')
#plt.scatter(x,y, c=color)
plt.scatter(df_iris['SepalLengthCm'],df_iris['PetalLengthCm'], c=df_iris['y'])
plt.show() # como usamos a magic cell no inicio, esse comando não é obrigatório.


# Observando o gráfico acima podemos verificar que com apenas duas features do dataset é possível separar as classes (com uma apenas amostra como de exceção).  Esse tipo de feature é muito útil para construirmos nossos modelos de aprendizagem.

# **Usando funções plot e hist do matplotlib para compreender melhor os dados:**

# In[ ]:


# Nesse grafico podemos ver que nosso dataset é bastante balanceado
plt.title('Histograma das Classes - Cada Classe tem 50 ocorrências')
plt.hist(df_iris['y'])
plt.show() # Estou mantendo o comando apenas por questão de estética na saída.
 
plt.title('Histograma da Propriedade Sepal Length\n Maior número de amostras com valor entre 5 e 7 cm')
plt.hist(df_iris['SepalLengthCm'], bins=6)
plt.show()


# In[ ]:


#Como as amostras estão ordenadas é possível "ver" no gráfico onde começa e termina
#cada grupo de 50.
plt.figure(figsize=(15,10))
plt.title('Exibindo as medidas por amostras')
plt.plot ( df_iris['SepalLengthCm'], c='blue', ) 
plt.plot ( df_iris['SepalWidthCm'], c= 'red')
plt.plot ( df_iris['PetalLengthCm'], c= 'green')
plt.plot ( df_iris['PetalWidthCm'], c= 'yellow')
plt.show()


# ##  5 - Selecionando um algoritmo de ML
# Embora seja um Dataset bem pequeno, ele é bastande balanceadol.  Vimos que as duas features  SepalLengthCm e PetalLengthCm sozinhas praticamente conseguem definir a separação das três classes mas queremos um classificador que faça o maior acerto possíve por isto vamos usar as 4 features que dispomos (O campo **Id ** será descartado para análises).  A biblioteca sckit-learn é muito útil para preparação de dados e para algoritmos que não envolvam redes neurais.
# 
# Com dados estruturados os modelos baseados em Regressão, Decisiona Tree e Random Forests são mais indicados. Mas qui vou somente por questão de didática vamos usar uma rede neural de 3 neurônios de com 1 saída (uma para cada tipo de flor) com 4 entradas (uma para cada feature de entrada), em algorítmos de classificação o número de saídas deve ser igual ao número de classes - se não for uma classificação binária(0 ou 1, true ou false, etc).
# 
# Não se preocupe caso não compreenda completamente alguma parte do código, vamos explorar todos detalhes nos próximos Labs, o objetivo aqui é você ver etapas gerais de uma solução completa usando Keras e TensorFlow. 

# In[ ]:


#veja que ao importar o keras que é um wrapper, o TensorFlow será exibido como backend
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils


# ## 6 - Feature Engineering 
# 
# Nesta fase temos a seleção das features que vão compor nosso Modelo - e seu ajuste para compatibilidade com o formato de entrada do tipo de algorítm ML selecionado.  Nossa coluna **'y'**, por exemplo contém valores de 1 a 3, esses valores serão transformados para 0 a 2 e convertidos em três colunas no formato** One-Hot**( esse tipo de codificação será explicada nos próximos labs). Além disso em muitos casos vamos ter que normalizar os valores de entrada antes entregar para uma rede neural ou algum outro tipo de algoritmo de ML.

# In[ ]:


#Número de classes possíveis
n_classes = len(df_iris['Species'].unique()) # 3 classes

#Fazemos slice do Dataframe e as convertemos em matrizes do NumPy
x_full = np.array(df_iris.iloc[:, 1:5].values) # selecionamos as colunas de features e todas linhas
y_full = np.array(df_iris.iloc[:, 6].values) # selecionamos todas linhas mas apenas a coluna 'y' 

# Para algorítimos de classificação com mais de duas classes temos que usar one-hot
# aqui uso uma simples subtração para alterar os valores de todas a linhas y
y_full = np_utils.to_categorical(y_full - 1, n_classes) 

print("Vericamos se as matrizes de entrada possuem o formato correto")
print("x_full.shape ={}    y_full.shape ={}".format(x_full.shape, y_full.shape))


# ## 7 - Split do dataset em treino e validação

# In[ ]:


seed = 42 # aqui ficamos o seed randômico, para garantir a reprodução de resultados

# A separação do dataset é uma técnica muito importante para maior eficiência
# da validação da eficácia de um modelo e veremos em maior detalhe nos próximos labs. 
X_train, X_val, y_train, y_val = train_test_split(x_full, y_full,
                                                test_size=0.2, random_state=seed)

# A classe train_test_split faz o embaralhamento dos dados antes
print("Novamente validamos os formatos do split:")
print(X_train.shape, y_train.shape); print(X_val.shape, y_val.shape)


# ## 8 - Definindo a arquitetura de nossa Rede Neural

# In[ ]:


# Fixamos o seed para biblioteca randômica do NumPy
np.random.seed(seed) 

#Cada implementação de um algorítimo de ML chamamos de modelo
model = Sequential()  # modelo sequencial
model.add(Dense(n_classes, input_shape=(4,))) # cria uma camada com 3 neurônios
model.add(Activation('softmax')) # usamos ativação de threshold conhecida como softmax
model.summary()


# Se olharmos acima notamos a existência de 15 parâmetros treináveis. Cada um dos três neurônios possuem 4 entradas, uma para cada feature, portanto teremos 4 x 3  = 12, ou seja 12 pesos que devem ser treinados. E de onde vem os 15?  
# 
# Lembra que para cada neurônio vamos ter uma entrada $x_{0}$ igual a 1 e um peso peso $w_{0}$ que será seu **bias**?   Então como temos 3 neurônios teremos 3 biases a serem treinados. Dai 12 pesos + 3 biases, resultando em 15 parâmetros treináveis.

# ## 9 - Compilando e Treinando nosso Modelo (Finalmente!!)

# In[ ]:


import timeit

n_epoch = 500 # Número de Épocas
batch_size = 10 #tamanho do Batch (quantidade de amostras por lote de treino)

#Aqui vamos usar o Gradiente Descendente Estocástico, que é um tipo de otimizador
sgd = SGD(lr= 0.1) # lr é o Learning Rate conceito que vamos ver nos próximos labs.

#Todo modelo precisa ser compilado, veja que no parâmetro loss informamos a função de erro
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy']) 

#Inicia contagem do tempo
start = timeit.default_timer()

# Aqui fazemos o fit do modelo e salvamos o resultado de cada epoca em history
history = model.fit(X_train, y_train, batch_size=batch_size, epochs=n_epoch,verbose=0) 

#Inicia contagem do tempo
elapsed = timeit.default_timer() - start

print("Rede Treinada em {} épocas durante {:.4f} segundos".format(n_epoch, elapsed))


# ## 10 - Avaliando o quão inteligente é nosso algorítimo

# In[ ]:


_, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
_, val_accuracy = model.evaluate(X_val, y_val, verbose=0)

print('Acurácia no Treino: {:.2f}%'.format(train_accuracy * 100))
print('Acurácia na Validação: {:.2f}%'.format(val_accuracy * 100))


# A acurácia é uma métrica que indica em percentual quantas amostras do todo foram classificadas corretamente.  
# ***Acurária = Número de Acertos / Número de Testes***
# 
# Em nosso dataset de treino conseguimos 117 acertos em 120 amostras (ou testes); e 30 acertos em 30 amostras no conjunto de teste. Com isso temos uma acurácia de 97,5% no treino e de 100% no dataset de teste.
# 
# Uau!!! Um excelente resultado com cerca de 5 segundos de treino e uma rede de somente 3 neurônios e um mínimo ajuste de **hiperparâmetro**, o **Learning Rate** (LR=0.1). 
# 
# Hiperparâmetros serão tema para um próximo lab. Fiquem a vontade para fazer Fork desse Kernel e testar valores diferentes para número de épocas, batch size e tipo de otimizador.  

# ** Matrix de Confusão** 
# 
# Esta é uma outra forma de visualizar a acurácia de uma rede, geralmente aplicamos somente no dataset de validação.
# Aqui apliquei nos dois para poder exibir onde nosso algoritmo errou.

# In[ ]:


y_hat_train = model.predict_classes(X_train)
pd.crosstab(y_hat_train, np.argmax(y_train, axis=1)) 


# In[ ]:


y_hat_val = model.predict_classes(X_val)
pd.crosstab(y_hat_val, np.argmax(y_val, axis=1))


# Se olharmos as duas matrizes de confusão veremos que nosso modelo errou apenas 3 amostras das 150. Um ótimo feito para uma rede de penas uma camada densa de 3 neurônios.

# ## 11 - Verificando a curva de aprendizagem de sua rede
# É possível verificar que a partir da época 300 não há grande melhoria (diminuição do erro)

# In[ ]:


# a impressão dos valores de perda a cada época de treinamento
# permite ter valiosos insights sobre como seu modelo se comporta durante o treinamento
plt.figure(figsize=(10,8))
plt.plot(history.history['loss'], label='Erro')
plt.plot(history.history['acc'], label='Acurácia')
plt.legend(loc='upper center')
plt.show()


# ## Tarefas do Lab
# 
# Crie um fork deste notebook em sua conta (assim você trabalha em sua própria cópia), e nos quadros abaixo escreva código para carregar o dataset  **House Sales in King County** (kc_house_data.csv) que já está copiado na pasta **../input/housesalesprediction** .  Se necessário crie novas células com Alt + Enter.
# 

# ### 1 - Carregar o Dataset House Sales de King County

# In[ ]:


print('Arquivo House Sales:')
print(check_output(["ls", "../input/housesalesprediction"]).decode("utf8"))


# In[ ]:


#Usando pandas crie um dataframe para armazenar o dataset House Sales
#df_house = pd.read_csv...


# ### 2 - Exibir as 20 primeiras linhas e as últimas 5 do data frame df_house

# In[ ]:


#Crie seu código abaixo


# ### 3 - Adicione uma nova coluna no dataframe com o nome 'yearsale' (o campo date possui a data da venda)

# In[ ]:


#


# ### 4 - Criar um gráfico que relacione o ano de cosntrução (yr_built) com o valor da venda (price)
# Aqui é possível utilizar a função plot ou scatter, veja qual funciona melhor.

# In[ ]:


#


# ### 5 - Mostrar o histograma de distribuição das vendas de acordo com o local (zipcode), preço de venda (price) e tamanho das casas (sqft_living)

# ### 6 - Avaliando de forma geral o conteúdo deste dataset qual ou quais colunas você acredita que tenha maior impacto sobre o valor da venda do imóvel?  Correlacione essas colunas com a coluna price. Plote gráficos que justifiquem sua resposta.

# ### 7 - Em ML recorremos ao conceito estatístico *Outlier*. Dada uma série de dados uma amostra que possua um valor muito destoante do restante é considerado um *outlier*.   Em algumas análises reconhecer outliers pode ser de grande ajuda para entender a natureza dos dados a serem explorados.  Como você faria para identificar a existência de outliers ao verificarmos o valor das vendas deste dataset?  Dica tente usar gráficos scatter e hist.  

# ### 8 - Usando Python e NumPy calcule o valor médio do square feet (pode utilizar a coluna sqft_living para o cálculo) e crie um gráfico para exibir todas as amostras cujo valor do square feet de venda seja maior que o valor médio.
