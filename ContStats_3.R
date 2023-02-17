library(mnormt)

##### QUESTION 1  #####

#   Initialise inputs
mu1 <- c(0, 0)
mu2 <- c(0, 0)
Cov1 <- cbind(c(1,0.8), c(0.8, 1))
Cov2 <- cbind(c(1, -0.7), c(-0.7,1))

### QUESTION 1 PART A ###

#   Generate data
set.seed(123) # for reproducibility
Xtrain1 <- rmnorm(n = 50, mean = mu1, Cov1, sqrt=NULL) 
set.seed(123) # for reproducibility
Xtrain2 <- rmnorm(n = 50, mean = mu2, Cov2, sqrt=NULL)
Xtrain <- rbind(Xtrain1, Xtrain2)

G1 <- rep(1,50)
G2 <- rep(2,50)
G <- c(G1,G2)
df <- data.frame(Xtrain = Xtrain, labels = G)

### QUESTION 1 PART B  ###

#   Extract parameters
ind1 = which(df$labels == 1)
ind2 = which(df$labels == 2)

N1 = length(ind1)
N2 = length(ind2)
N = N1 + N2

pi_hat_1 <- N1/N
pi_hat_2 <- N2/N

mu_hat_1 <- c(mean(Xtrain[ind1,1]), mean(Xtrain[ind1,2]))
mu_hat_2 <- c(mean(Xtrain[ind2,1]), mean(Xtrain[ind2,2]))

Cov_hat_1 <- cov(Xtrain[ind1,])
Cov_hat_2 <- cov(Xtrain[ind2,])

Cov_hat_LDA <- (1/(N - 2)) * ((N1 - 1) * Cov_hat_1 + (N2 - 1) * Cov_hat_2)

#   Define discriminant function
#   Can be used for LDA or QDA dependeing on choice of Covariance argument
discriminant <- function(x,pi,mu,Cov){
  log(pi) - (1 / 2) * log(det(Cov)) - (1 / 2) * (x - mu_hat_1) %*% inv(Cov) %*% (x- mu)
}

### QUESTION 1 PART C  ###

#   Generate test data
set.seed(123) # for reproducibility
Xtest1 <- rmnorm(n = 50, mean = mu1, Cov1, sqrt=NULL) 
set.seed(123)   # for reproducibility
Xtest2 <- rmnorm(n = 50, mean = mu2, Cov2, sqrt=NULL)
Xtest <- rbind(Xtest1, Xtest2)
G1_test <- rep(1,N1)
G2_test <- rep(2,N2)
G_test <- c(G1,G2)


#   Predict using LDA
d1 = 0
d2 = 0 
G_hat_LDA = rep(0,N)
for(i in 1:N){
  d1 = discriminant(Xtest[i,],pi_hat_1,mu_hat_1,Cov_hat_LDA)
  d2 = discriminant(Xtest[i,],pi_hat_2,mu_hat_2,Cov_hat_LDA)
  G_hat_LDA[i] = which.max(c(d1,d2))
}

#   Predict using QDA
d1 = 0
d2 = 0 
G_hat_QDA = 1:N
for(i in 1:N){
  d1 = discriminant(Xtest[i,],pi_hat_1,mu_hat_1,Cov_hat_1)
  d2 = discriminant(Xtest[i,],pi_hat_2,mu_hat_2,Cov_hat_2)
  G_hat_QDA[i] = which.max(c(d1,d2))
}

#   Calculate the misclassification rate for LDA
pred_LDA <- rep(0,N)
for(i in 1:N){
  if(G_test[i] == G_hat_LDA[i]){
    pred_LDA[i] <- 1
  }
  else{
    pred_LDA[i] <- 0
  }
}
misclass_rate_LDA <- 1 - sum(pred_LDA) / N

#   Calculate the misclassification rate for QDA
pred_QDA <- rep(0,N)
for(i in 1:N){
  if(G_test[i] == G_hat_QDA[i]){
    pred_QDA[i] <- 1
  }
  else{
    pred_QDA[i] <- 0
  }
}

misclass_rate_QDA <- 1 - sum(pred_QDA) / N

print(c(misclass_rate_LDA,misclass_rate_QDA))

