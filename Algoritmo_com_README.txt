

############################################### NAIVE BAYES COM BAG OF WORDS ############################################################
#Carregar pacotes

library(tm)
library(e1071)
library(dplyr)
library(caret)


# Abrindo o banco de dados em formato csv
BDARROZMAR2019 <- read.csv2("BDARROZMAR2019_60GtinArrozValido.csv")
str(BDARROZMAR2019)

#Definindo o data frame de trabalho
df <- BDARROZMAR2019[,c("Descricao_Produto","Cod_GTIN2")] %>% 
          rename("GTIN"="Cod_GTIN2") %>% 
            rename("Descricao"="Descricao_Produto")

#Apagando banco completo
rm(BDARROZMAR2019)

#Limpando a base de dados
df$Descricao %<>% iconv(to = "ASCII//TRANSLIT") %>% #removendo os acentos
                        removePunctuation() %>% #removendo pontuacao
                          toupper() #colocando tudo em MAIUCULA


df$Descricao <- gsub(pattern = "TIPO1 KG", replacement = "TIPO1 1KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP-1", replacement = "TP1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP 1", replacement = "TP1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP 2", replacement = "TP2", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIPO 1", replacement = "TIPO1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIPO 2", replacement = "TIPO2", x = df$Descricao)
df$Descricao <- gsub(pattern = "1 KG", replacement = "1KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "5 KG", replacement = "5KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIO MINGOTE1", replacement = "TIO MINGOTE 1", x = df$Descricao)

#separando um banco de dados sem repeticoes de registros baseadas no GTIN e na descricao
#So executar quando for trabalhar com a base completa considerando repetição
length(unique(df$Descricao)) #contando o numero de descricao distintas
df <- unique(df)
nrow(df) #contanto o numero de registros distintos

#Embaralhando os dados
set.seed(1)
df <- df[sample(nrow(df),replace = FALSE), ]
glimpse(df)

#Transformando o GTIN em fator

df$GTIN <- as.factor(df$GTIN)
class(df$GTIN)

#Transformando em Corpus tokenizado

corpus <- Corpus(VectorSource(df$Descricao));
inspect(corpus[1:3])

#Retirando os espacos extras
corpus.clean <- corpus %>%  tm_map(stripWhitespace)
inspect(corpus.clean[1:3])

#Seleção dos folds da validação cruzada

folds = createFolds(df$GTIN, k = 2)
Data=NULL
folds
i=0
for (i in 1:as.numeric(length(folds))){

sink("ajust_nb_bag_k2.log") #Guandadndo os resultados

# Particionando as bases em treinamento e teste

df.train <- df[-(folds[[i]]), ]
df.test <- df[(folds[[i]]), ]
nrow(df) == nrow(df.train)+nrow(df.test)

corpus.clean.train <- corpus.clean[-(folds[[i]])]
corpus.clean.test <- corpus.clean[(folds[[i]])]

# Montando a Matriz de termos do documento com modelo de Bag of Word #####

# Selecionando token com frequencias maior que 5
dtm <- DocumentTermMatrix(corpus.clean)
dtm.train <- dtm[-(folds[[i]]), ]
fivefreq <- findFreqTerms(dtm.train, 5)
length((fivefreq))
rm(dtm.train)
rm(dtm)

# Usando somente as 5 palabras mais frequentes para criar a DTM
dtm.train.nb <- DocumentTermMatrix(corpus.clean.train, control=list(dictionary = fivefreq))
dim(dtm.train.nb)
dtm.test.nb <- DocumentTermMatrix(corpus.clean.test, control=list(dictionary = fivefreq))
dim(dtm.test.nb)

# Definindo funcao para converter as frequencias das palavras em rotulos sim (presente) e nao (ausencia)
convert_count <- function(x) { y <- ifelse(x > 0, "Yes","No") }

# Aplicando a conversao das contagem nos DTM de treino e de test
trainNB <- apply(dtm.train.nb, 2, convert_count)
testNB <- apply(dtm.test.nb, 2, convert_count)

# Treinando o classificador do modelo Naive Bayes
system.time( classifier <- naiveBayes(trainNB, df.train$GTIN, laplace = 1))

#Testando a predicao do modelo aplicando o classificador no conjunto de teste
system.time( pred <- predict(classifier, newdata=testNB) )

# Prepare the confusion matrix
conf.mat <- confusionMatrix(pred, df.test$GTIN)

Data<-rbind(Data,cbind(i,conf.mat$overall[1]))
cat(paste("Calma, ja estou no step",i))
i=i+1
}
Data #printando os resultados
mean(Data[,2]) #Calculo da média das acurácias
sd(Data[,2])/mean(Data[,2]) #Calculo do coeficiente de variação

sink() #Fechando o output

############################################### NAIVE BAYES COM TF-IDF ############################################################

#Carregar pacotes

library(tm)
library(e1071)
library(dplyr)
library(caret)

# Abrindo o banco de dados em formato csv
BDARROZMAR2019 <- read.csv2("BDARROZMAR2019_60GtinArrozValido.csv")
str(BDARROZMAR2019)

#Definindo o data frame de trabalho
df <- BDARROZMAR2019[,c("Descricao_Produto","Cod_GTIN2")] %>% 
  rename("GTIN"="Cod_GTIN2") %>% 
  rename("Descricao"="Descricao_Produto")

#Apagando banco completo
rm(BDARROZMAR2019)

#Limpando a base de dados
df$Descricao %<>% iconv(to = "ASCII//TRANSLIT") %>% #removendo os acentos
  removePunctuation() %>% #removendo pontuacao
  toupper() #colocando tudo em MAIUSCULA

df$Descricao <- gsub(pattern = "TIPO1 KG", replacement = "TIPO1 1KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP-1", replacement = "TP1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP 1", replacement = "TP1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP 2", replacement = "TP2", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIPO 1", replacement = "TIPO1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIPO 2", replacement = "TIPO2", x = df$Descricao)
df$Descricao <- gsub(pattern = "1 KG", replacement = "1KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "5 KG", replacement = "5KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIO MINGOTE1", replacement = "TIO MINGOTE 1", x = df$Descricao)

#separando um banco de dados sem repeticoes de registros baseadas no GTIN e na descricao
#So executar quando for trabalhar com a base completa considerando repetição
length(unique(df$Descricao)) #contando o numero de descricao distintas
df <- unique(df)
nrow(df) #contanto o numero de registros distintos

#Embaralhando os dados
set.seed(1)
df <- df[sample(nrow(df),replace = FALSE), ]
glimpse(df)

#Transformando o GTIN em fator
df$GTIN <- as.factor(df$GTIN)
class(df$GTIN)

#Transformando em Corpus tokenizado

corpus <- Corpus(VectorSource(df$Descricao));
inspect(corpus[1:3])

#Retirando os espacos extras
corpus.clean <- corpus %>%  tm_map(stripWhitespace)
inspect(corpus.clean[1:3])

dtm <- DocumentTermMatrix(corpus.clean,control=list(weighting = function(x) weightTfIdf(x, normalize = FALSE)))



sink("ajust_tfidf_k2.log")#Guandando os resultados


#Seleção dos folds da validação cruzada

folds = createFolds(df$GTIN, k = 2)
Data=NULL
i=0
for (i in 1:as.numeric(length(folds))){


#Separando uma parte daos dados para treinamento e outra para teste considerando o número de folds

dtm_train<-dtm[-folds[[i]],]
dtm_test<-dtm[folds[[i]],]
nrow(dtm_train)
nrow(dtm_test)

Data_train_labels <- df[-folds[[i]], ]$GTIN
Data_test_labels  <- df[folds[[i]], ]$GTIN

convert_counts <- function(x) { x <- ifelse(x > 0, "Yes", "No")}

# apply() convert_counts() to columns of train/test data
Data_train <- apply(dtm_train, MARGIN = 2, convert_counts)
Data_test  <- apply(dtm_test, MARGIN = 2, convert_counts)

#Aplicando o algoritmo de classificação

classifier <- naiveBayes(Data_train, Data_train_labels,laplace = 1)

#Predição baseada no modelo de classificação

pred <- predict(classifier, Data_test)

# Criando uma matrix de confusão
conf.mat<-confusionMatrix(pred,Data_test_labels)

#Guardadndo os resultados de cada folds

Data<-rbind(Data,cbind(i,conf.mat$overall[1]))
cat(paste("Calma, ja estou no step",i) )
i=i+1
}

Data #printando os resultados
mean(Data[,2]) #Calculo da média das acurácias
sd(Data[,2])/mean(Data[,2]) #Calculo do coeficiente de variação

sink()#Fechando o output

############################################### SUPPORT VECTOR MACHINE ###########################################################################

#Carregar pacotes

library(tm)
library(e1071)
library(dplyr)
library(caret)


# Abrindo o banco de dados em formato csv
BDARROZMAR2019 <- read.csv2("BDARROZMAR2019_60GtinArrozValido.csv")
str(BDARROZMAR2019)

#Definindo o data frame de trabalho
df <- BDARROZMAR2019[,c("Descricao_Produto","Cod_GTIN2")] %>% 
  rename("GTIN"="Cod_GTIN2") %>% 
  rename("Descricao"="Descricao_Produto")

#Apagando banco completo
rm(BDARROZMAR2019)

#Limpando a base de dados
df$Descricao %<>% iconv(to = "ASCII//TRANSLIT") %>% #removendo os acentos
  removePunctuation() %>% #removendo pontuacao
  toupper() #colocando tudo em MAIUSCULA


df$Descricao <- gsub(pattern = "TIPO1 KG", replacement = "TIPO1 1KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP-1", replacement = "TP1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP 1", replacement = "TP1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP 2", replacement = "TP2", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIPO 1", replacement = "TIPO1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIPO 2", replacement = "TIPO2", x = df$Descricao)
df$Descricao <- gsub(pattern = "1 KG", replacement = "1KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "5 KG", replacement = "5KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIO MINGOTE1", replacement = "TIO MINGOTE 1", x = df$Descricao)

#separando um banco de dados sem repeticoes de registros baseadas no GTIN e na descricao
#So executar quando for trabalhar com a base completa considerando repetição
length(unique(df$Descricao)) #contando o numero de descricao distintas
df <- unique(df)
nrow(df) #contanto o numero de registros distintos

#Embaralhando os dados
set.seed(1)
df <- df[sample(nrow(df),replace = FALSE), ]
glimpse(df)

#Transformando o GTIN emfator

df$GTIN <- as.factor(df$GTIN)
class(df$GTIN)

#Transformando em Corpus tokenizado

corpus <- Corpus(VectorSource(df$Descricao));
inspect(corpus[1:3])

#Retirando os espacos extras
corpus.clean <- corpus %>%  tm_map(stripWhitespace)
inspect(corpus.clean[1:3])

#Criando a matriz de termos dos documentos

dtm <- DocumentTermMatrix(corpus.clean)

#Transformando a dtm em um dataframe

df <- cbind(df, as.matrix(dtm)); df <- df[,-1]

sink("ajust_svm_cost1000_k2.log") #Guandando os resultados


#Seleção dos folds da validação cruzada

folds = createFolds(df$GTIN, k = 2)
Data=NULL
i=0

for (i in 1:as.numeric(length(folds))){

#Separando uma parte daos dados para treinamento e outra para teste considerando o número de folds

train<-df[-folds[[i]],]
test<-df[folds[[i]],]
nrow(train)
nrow(test)

#Aplicando o algoritmo de classificação

classifier <- svm(formula = GTIN ~ .,data = (train),  kernel = 'radial',scale = F,cost=1000)

#Predição baseada no modelo de classificação

pred <- predict(classifier, test)

# Criando uma matrix de confusão

conf.mat<-confusionMatrix(table(pred,test$GTIN))

#Guardadndo os resultados de cada folds

Data<-rbind(Data,cbind(i,conf.mat$overall[1]))
cat(paste("Calma, ja estou no step",i) )
i=i+1
}

Data #printando os resultados
mean(Data[,2]) #Calculo da média das acurácias
sd(Data[,2])/mean(Data[,2]) #Calculo do coeficiente de variação

sink()#Fechando o output

############################################### ARVORE DE DECISÃO ###########################################################################

#Carregar pacotes

library(rpart)
library(tm)
library(e1071)
library(dplyr)
library(caret)
library(magrittr)

# Abrindo o banco de dados em formato csv
BDARROZMAR2019 <- read.csv2("BDARROZMAR2019_60GtinArrozValido.csv")
str(BDARROZMAR2019)

#Definindo o data frame de trabalho
df <- BDARROZMAR2019[,c("Descricao_Produto","Cod_GTIN2")] %>% 
  rename("GTIN"="Cod_GTIN2") %>% 
  rename("Descricao"="Descricao_Produto")

#Apagando banco completo
rm(BDARROZMAR2019)

#Limpando a base de dados
df$Descricao %<>% iconv(to = "ASCII//TRANSLIT") %>% #removendo os acentos
  removePunctuation() %>% #removendo pontuacao
  toupper() #colocando tudo em MAIUCULA


df$Descricao <- gsub(pattern = "TIPO1 KG", replacement = "TIPO1 1KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP-1", replacement = "TP1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP 1", replacement = "TP1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TP 2", replacement = "TP2", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIPO 1", replacement = "TIPO1", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIPO 2", replacement = "TIPO2", x = df$Descricao)
df$Descricao <- gsub(pattern = "1 KG", replacement = "1KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "5 KG", replacement = "5KG", x = df$Descricao)
df$Descricao <- gsub(pattern = "TIO MINGOTE1", replacement = "TIO MINGOTE 1", x = df$Descricao)

#separando um banco de dados sem repeticoes de registros baseadas no GTIN e na descricao
#So executar quando for trabalhar com a base completa considerando repetição
length(unique(df$Descricao)) #contando o numero de descricao distintas
df <- unique(df)
nrow(df) #contanto o numero de registros distintos

#Embaralhando os dados
set.seed(1)
df <- df[sample(nrow(df),replace = FALSE), ]
glimpse(df)

#Transformando o GTIN em fator
df$GTIN <- as.factor(df$GTIN)
class(df$GTIN)

#Transformando os dados em Corpus tokenizado

corpus <- Corpus(VectorSource(df$Descricao));
inspect(corpus[1:3])

#Retirando os espacos extras
corpus.clean <- corpus %>%  tm_map(stripWhitespace)
inspect(corpus.clean[1:3])

#Criando a matriz de termos dos documentos

dtm <- DocumentTermMatrix(corpus.clean)

#Transformando a dtm em um dataframe

df <- cbind(df, as.matrix(dtm)); df <- df[,-1]

sink("ajust_arvore_k2.log") #Guandadndo os resultados

#Seleção dos folds da validação cruzada

folds = createFolds(df$GTIN, k = 2)
Data=NULL
i=0

for (i in 1:as.numeric(length(folds))){

#Separando uma parte daos dados para treinamento e outra para teste considerando o número de folds

train<-df[-folds[[i]],]
test<-df[folds[[i]],]
nrow(train)
nrow(test)

#Aplicando o algoritmo de classificação

classifier <- classifier <- rpart(GTIN~., data=train,method = "class")


#Predição baseada no modelo de classificação

pred <- predict(classifier, test, type="class")

# Criando uma matrix de confusão

conf.mat<-confusionMatrix(table(pred,test$GTIN))

#Guardando os resultados de cada folds

Data<-rbind(Data,cbind(i,conf.mat$overall[1]))
cat(paste("Calma, ja estou no step",i) )
i=i+1
}

Data #printando os resultados
mean(Data[,2]) #Calculo da média das acurácias
sd(Data[,2])/mean(Data[,2]) #Calculo do coeficiente de variação

sink() #Fechando o output

