install.packages("titanic")
install.packages('imputation')
install.packages('dplyr')
install.packages('FactoMineR')
install.packages('MASS')
install.packages('class')
install.packages('createDataPartition')
library(titanic)
library(VIM)
library(ggplot2) 
library(dplyr)
library(caret)
library(MASS)
library(class)


#######   Contemporary Statistics Project   #######


###   Load data set
data_train = titanic_train
data_test = titanic_test

### Imputation of missing values
data_train_complete <- kNN(data_train ,k = 5) %>% subset(select = -c(13:24)) 
data_test_complete <- kNN(data_test ,k = 5)   %>% subset(select = -c(12:22))


#####   DATA VISUALISATION AND EXPLORATION    #####

###   compute some summary statistics
sex_group <- data_train_complete %>% group_by(Sex)  %>%
  summarise(Total = length(Sex),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

Embarked_group <- data_train_complete %>% group_by(Embarked)  %>%
  summarise(Total = length(Embarked),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

Class_group <- data_train_complete %>% group_by(Pclass)  %>%
  summarise(Total = length(Pclass),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

SibSp_group <- data_train_complete %>% group_by(SibSp)  %>%
  summarise(Total = length(SibSp),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

Parch_group <- data_train_complete %>% group_by(Parch)  %>%
  summarise(Total = length(Parch),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

hist(data_train_complete$Age)   #   visualise age and Fare counts
hist(data_train_complete$Fare)

Survived_group <- data_train_complete %>% group_by(Survived)  %>% #check the mean age and fare
  summarise(Total = length(SibSp),
            Mean_Age = mean(Age),
            Median_Fare = median(Fare),
            .groups = 'drop')


### Prepare the data for PCA by removing irrelevant features, and formatting category data as numeric
train_features <- subset(data_train_complete, select = -c(PassengerId, Name, Cabin, Ticket, Survived))  #subtract features
test_features <- subset(data_test_complete, select = -c(PassengerId, Name, Cabin, Ticket))

train_features$Sex <- factor(train_features$Sex)  # formatting
test_features$Sex <- factor(test_features$Sex)
train_features$Embarked <- factor(train_features$Embarked)
test_features$Embarked <- factor(test_features$Embarked)


train_features[, c('Sex', 'Embarked')] <- sapply(train_features[, c('Sex', 'Embarked')], as.numeric) # represent categorical data as numeric
test_features[, c('Sex', 'Embarked')] <- sapply(test_features[, c('Sex', 'Embarked')], as.numeric)


#####   PRINCIPAL COMPONENT ANALYSIS    #####

###   function to compute principal components
myPCA <- function(A){
  A_scaled <- scale(A)  # 1.  scale and centre the data
  C <- cov(A_scaled)    # 2.  compute the covariance matrix
  A_eigen <- eigen(C)   # 3.  extract eigenvalues and eigenvectors
  pc <- A_eigen$vectors #       principal components are the eigenvectors of the cov matrix
  explained_variance <- cumsum(A_eigen$values)/sum(A_eigen$values)  # calculate how much of the variance is explained by a given number of components
  print(explained_variance)
  return(pc)
}

###   Format and scale the data
A_train = as.matrix(train_features)
A_train_scaled <- scale(A_train)
A_test <- as.matrix(test_features)  
A_train_means <- colMeans(A_train)
A_test_scaled <- scale(A_test, center= A_train_means) #   we center the test data using the means of training data


pc = myPCA(A_train) #compute principal components

###   Transform training and test data using top 5 principal components
Z_train = A_train_scaled%*%pc[,1:5]
Z_test = A_test_scaled%*%pc[,1:5]
Z_train.df <- data.frame(Z_train)
Z_test.df <- data.frame(Z_test)



###     CLASSIFY USING LDA    #####

library(matlib)

### Function to predict using LDA on two classes
myLDA <- function(Xtrain,Xtest,labels){
  #   Extract parameters
  ind1 = which(labels == 0)   #   find class indices
  ind2 = which(labels == 1)
  
  N1 = length(ind1) #class counts
  N2 = length(ind2)
  N = N1 + N2
  
  pi_hat_1 <- N1/N #   class ratios
  pi_hat_2 <- N2/N
  
  mu_hat_1 <- colMeans(Xtrain[ind1,]) #   class means
  mu_hat_2 <- colMeans(Xtrain[ind2,])
  
  Cov_hat_1 <- cov(Xtrain[ind1,]) #   classwise covariance
  Cov_hat_2 <- cov(Xtrain[ind2,])
  
  Cov_hat_LDA <- (1/(N - 2)) * ((N1 - 1) * Cov_hat_1 + (N2 - 1) * Cov_hat_2) # covariance argument for LDA
  
  #   Define discriminant function
  #   Can be used for LDA or QDA depending on choice of Covariance argument
  discriminant <- function(x,pi,mu,Cov){
    log(pi) - (1 / 2) * log(det(Cov)) - (1 / 2) * (x - mu_hat_1) %*% inv(Cov) %*% (x- mu)
  }
  
  #   Predict using LDA
  d1 = 0    #   initialise 
  d2 = 0 
  M <- length(Xtest[,1])
  G_hat_LDA = rep(0,M)
  for(i in 1:M){
    d1 = discriminant(Xtest[i,],pi_hat_1,mu_hat_1,Cov_hat_LDA) #    compute discriminant for each and assign to class which maximises it
    d2 = discriminant(Xtest[i,],pi_hat_2,mu_hat_2,Cov_hat_LDA)
    G_hat_LDA[i] = which.max(c(d1,d2))-1
  }
  return(G_hat_LDA)
}

###   function to test LDA predictions using Cross Validation
myCrossValLDA <- function(X, labels, k){
  c <- c(1:length(X[,1])) #   vector of indices    
  partition <- createFolds(c, k, list = TRUE, returnTrain = FALSE) #    create a partition
  train_error <- rep(0,k) #   initialise vectors
  test_error <- rep(0,k)
  for (i in 1:k){
    idxTrain <- unlist(partition[-i]) #   iterate over partitioned data
    idxTest  <- unlist(partition[i])
    trainlabs <- array(labels[idxTrain])
    testlabs <- array(labels[idxTest])
    D_train <- X[idxTrain,]
    D_test <- X[idxTest,]
    
    
    prediction_train <- myLDA(D_train, D_train, trainlabs) #  predict classes for partitioned data
    prediction_test <- myLDA(D_train, D_test, trainlabs)
    
    train_error[i] <- 1 - mean(prediction_train==trainlabs) #   calculate training and test error
    test_error[i] <- 1 - mean(prediction_test==testlabs)
  }
  train_error <- mean(train_error) #    compute average training and test error
  test_error <- mean(test_error)
  results = c(train_error, test_error)
  return(results)
}

misclass_rate_LDA <- myCrossValLDA(Z_train, data_train_complete$Survived, 10) # save errors from cross validation


pred_LDA <- myLDA(Z_train,Z_test,data_train_complete$Survived)


#####   CLASSIFY USING QDA    #####

### function to predict using QDA on 2 classes
myQDA <- function(Xtrain,Xtest,labels){
  #   Extract parameters
  ind1 = which(labels == 0) #   indices for each class
  ind2 = which(labels == 1)
  
  N1 = length(ind1) # class counts
  N2 = length(ind2)
  N = N1 + N2
  
  pi_hat_1 <- N1/N #    proportion in each class
  pi_hat_2 <- N2/N 
  
  mu_hat_1 <- colMeans(Xtrain[ind1,]) # class means
  mu_hat_2 <- colMeans(Xtrain[ind2,])
  
  Cov_hat_1 <- cov(Xtrain[ind1,]) #   covariance matrix for each class
  Cov_hat_2 <- cov(Xtrain[ind2,])

  #   Define discriminant function
  #   Can be used for LDA or QDA depending on choice of Covariance argument
  discriminant <- function(x,pi,mu,Cov){
    log(pi) - (1 / 2) * log(det(Cov)) - (1 / 2) * (x - mu_hat_1) %*% inv(Cov) %*% (x- mu)
  }
  
  #   Predict using QDA
  d1 = 0 # initialise
  d2 = 0 
  M = length(Xtest[,1])
  G_hat_QDA = rep(0,M)
  for(i in 1:M){
    d1 = discriminant(Xtest[i,],pi_hat_1,mu_hat_1,Cov_hat_1) #    calculate discriminant for each class and choose the maximum
    d2 = discriminant(Xtest[i,],pi_hat_2,mu_hat_2,Cov_hat_2) 
    G_hat_QDA[i] = which.max(c(d1,d2))-1
  }
  return(G_hat_QDA) # returns class labels
}

###   test model using cross validation
myCrossValQDA <- function(X, labels, k){
  c <- c(1:length(X[,1])) # indices
  partition <- createFolds(c, k, list = TRUE, returnTrain = FALSE) #    partition the data set
  train_error <- rep(0,k) # initialise
  test_error <- rep(0,k)
  for (i in 1:k){
    idxTrain <- unlist(partition[-i]) #   use the splits as training and test data
    idxTest  <- unlist(partition[i])
    trainlabs <- array(labels[idxTrain])
    testlabs <- array(labels[idxTest])
    D_train <- X[idxTrain,]
    D_test <- X[idxTest,]
    
    
    prediction_train <- myQDA(D_train, D_train, trainlabs) #    predict classes for training and test data
    prediction_test <- myQDA(D_train, D_test, trainlabs)
    
    train_error[i] <- 1 - mean(prediction_train==trainlabs) #   compute training and test error
    test_error[i] <- 1 - mean(prediction_test==testlabs)
  }
  train_error <- mean(train_error) # return average training and test error from each split
  test_error <- mean(test_error) 
  results = c(train_error, test_error)
  return(results)
}

misclass_rate_QDA <- myCrossValQDA(Z_train, data_train_complete$Survived, 10) # save results of cross validation

pred_QDA <- myQDA(Z_train,Z_test,data_train_complete$Survived) # predict using QDA


#####     CLASSIFY USING KNN      #####

###   test kNN classifier using cross validation
myCrossValkNN <- function(X, labels, k, K){
  c <- c(1:length(X[,1])) #   indices
  partition <- createFolds(c, k, list = TRUE, returnTrain = FALSE) #  partition data
  train_error <- rep(0,k)
  test_error <- rep(0,k)
  for (i in 1:k){
    idxTrain <- unlist(partition[-i]) # iterate over data splits as training and test data
    idxTest  <- unlist(partition[i])
    trainlabs <- array(labels[idxTrain])
    testlabs <- array(labels[idxTest])
    D_train <- X[idxTrain,]
    D_test <- X[idxTest,]
    
    prediction_train <- knn(D_train, D_train, trainlabs, K) %>% array() %>% sapply(as.numeric) #    predict using knn
    prediction_test <- knn(D_train, D_test, trainlabs, K) %>% array() %>% sapply(as.numeric)
    
    train_error[i] <- 1 - mean(prediction_train==trainlabs) #   compute training and test error
    test_error[i] <- 1 - mean(prediction_test==testlabs)
  }
  train_error <- mean(train_error) # return the mean training and test error over each split
  test_error <- mean(test_error)
  results = c(train_error, test_error)
  return(results)
}

### compute training and test errors for several valus of k to find the optimal value
misclass_rate <- cbind(rep(0,20),rep(0,20)) 
for (k in 1:20){
  misclass_rate[k,] <- myCrossValkNN(Z_train, data_train_complete$Survived, 10, k)
}

misclass_rate = data.frame(misclass_rate) # format for plotting

#   plot the training and test errors for each value of k
ggplot(misclass_rate, aes(x = 1:20)) +
  geom_line(data = misclass_rate, aes(y = X1, color = 'train'), show.legend = TRUE)+
  geom_line(data = misclass_rate, aes(y = X2, color = 'test'), show.legend = TRUE)+
  labs(x = 'k', y = 'Error')+
  scale_color_manual(name = 'Error', values = c('train' = 'red' ,'test' = 'blue'))

k_opt = which.min(misclass_rate$X2) #   save optimal k
misclass_rate_kNN <- myCrossValkNN(Z_train, data_train_complete$Survived, 10, k_opt) # compute error for optimal k

pred_knn <- prediction_train <- knn(Z_train, Z_test, data_train_complete$Survived, k_opt) %>% array() %>% sapply(as.numeric) #    predict using optimal k


######    Analyse the results
new_data <- cbind(data_test_complete,pred_knn) %>% rename(Survived = pred_knn)#   add predictions to the data

###   compute some summary statistics
sex_group_new <- new_data %>% group_by(Sex)  %>%
  summarise(Total = length(Sex),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

Embarked_group_new <- new_data %>% group_by(Embarked)  %>%
  summarise(Total = length(Embarked),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

Class_group_new <- new_data %>% group_by(Pclass)  %>%
  summarise(Total = length(Pclass),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

SibSp_group_new <- new_data %>% group_by(SibSp)  %>%
  summarise(Total = length(SibSp),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

Parch_group_new <- new_data %>% group_by(Parch)  %>%
  summarise(Total = length(Parch),
            Survived = sum(Survived),
            ratio = Survived/Total,
            .groups = 'drop')

hist(new_data$Age)   #   visualise age and Fare counts
hist(new_data$Fare)

Survived_group_new <- new_data %>% group_by(Survived)  %>% #check the mean age and fare
  summarise(Total = length(SibSp),
            Mean_Age = mean(Age),
            Median_Fare = median(Fare),
            .groups = 'drop')

