#R Packages for Gaussian Process Regresion 
#Afra Kilic
#March, 2023

#Libraries

library(mgcv)
library(brms)
library(kernlab)
library(GPFDA)
library(CVEK)
library(gam)
library(GauPro)

#Simulation study with one-predictor

#Data Generation
set.seed(123)
n <- 250
xobs <- sort(runif(n,min=-3,max=3))
mu = rep(0,length(xobs))
sigma2 <- .1**2
error <- rnorm(n,sd=sqrt(sigma2))
yobs <- .5*dnorm(xobs)*xobs + error
data = as.data.frame(cbind(yobs, xobs))

#splitting into train and test data 
sample <- sample(c(TRUE, FALSE), nrow(data), replace=TRUE, prob=c(0.8,0.2))
train <- data[sample,]
test <- data[!sample,]


#MODEL FITS AND PREDICTONS 

######################################################################################################
#LM fit 

start_time_lm1 <- Sys.time()
fit_lm1 <- lm(yobs~ xobs, data = train)
end_time_lm1 <- Sys.time()
computation_time_lm1 <- end_time_lm1 - start_time_lm1
cat("Computation time LM:", computation_time_lm1, "seconds")

summary(fit_lm1)

#Plot the 95CI 
df <- data.frame(x = train$xobs,
                 y = train$yobs)
mod <- lm(y ~ x, data = df)
# predicts + interval
newx <- seq(min(df$x), max(df$x), length.out=100)
preds <- predict(mod, newdata = data.frame(x=newx), 
                 interval = 'confidence')
# plot
plot(y ~ x, data = df, type = 'n', main = "LM fit (95CI)")
# add fill
polygon(c(rev(newx), newx), c(rev(preds[ ,3]), preds[ ,2]), col = 'grey80', border = NA)
# model
abline(mod)
# intervals
lines(newx, preds[ ,3], lty = 'dashed', col = 'red')
lines(newx, preds[ ,2], lty = 'dashed', col = 'red')


#Testing the model 
pred_lm1 <- predict(fit_lm1, newdata = test)

plot(test$xobs, test$yobs, main = "Actual Observations vs. Predictons", 
     xlab = "x", ylab = "y", col = "blue", pch = 19)
points(test$xobs, pred_lm1, col = "red", type = "b")
legend(-2, 0.3, legend=c("Actual", "Predictions"), col = c("blue", "red"), lty=1:2, cex=0.8)

######################################################################################################

#MGCV fit
start_time_gam1 <- Sys.time()
fit_gam1 <- mgcv::gam(yobs~ s(xobs), family='gaussian')
end_time_gam1 <- Sys.time()
computation_time_gam1 <- end_time_gam1 - start_time_gam1
cat("Computation time GAM:", computation_time_gam1, "seconds")
summary(fit_gam1)
plot(fit_gam1,pages=1,residuals=TRUE, main = "Standardized Residuals", xlab = "x")  ## show partial residuals

#Testing the model 
pred_gam1 <- predict(fit_gam1, newdata = test)


plot(test$xobs, test$yobs, main = "Actual Observations vs. Predictons", 
     xlab = "x", ylab = "y", col = "blue", pch = 19)
points(test$xobs, pred_gam1, col = "red", type = "b")
legend(-2, 0.3, legend=c("Actual", "Predictions"), col = c("blue", "red"), lty=1:2, cex=0.8)


######################################################################################################

#brms fit
set.seed(123)
start_time_brms1 <- Sys.time()
fit_brms1 <- brms::brm(yobs ~ gp(xobs), data = train, chains = 2)
end_time_brms1 <- Sys.time()
computation_time_brms1 <- end_time_brms1 - start_time_brms1
cat("Computation time BRMS:", computation_time_brms1, "seconds")
summary(fit_brms1)
#Testing the model 
pred_brms1 <- predict(fit_brms1, newdata = test)
plot(test$xobs, test$yobs, main = "Actual Observations vs. Predictons", 
     xlab = "x", ylab = "y", col = "blue", pch = 19, ylim=c(min(pred_brms1[,3]), max(pred_brms1[,4])))
points(test$xobs, pred_brms1[,1], col = "red", type = "b")
lines(test$xobs, pred_brms1[,3], col = "gray") #lower bound
lines(test$xobs, pred_brms1[,4], col = "gray")#upper bound
legend(-2, 0.3, legend=c("Actual", "Predictions", "CI95"), col = c("blue", "red", "gray"), lty=1:2, cex=0.8)

brms::pp_check(fit_brms1)
######################################################################################################

#CVEK fit
set.seed(123)
kern_par <- data.frame(method = c("linear", "rbf"), #model library specification
                       l = rep(1, 2), p = 1:2, stringsAsFactors = FALSE)
# define kernel library
kern_func_list <- CVEK::define_library(kern_par)

start_time_cvek1 <- Sys.time()
fit_CVEK1 <- CVEK::cvek(yobs ~  k(xobs) , kern_func_list = kern_func_list, data = train, 
                        lambda = exp(seq(-3, 5)), test = "asymp")   
end_time_cvek1 <- Sys.time()
computation_time_cvek1 <- end_time_cvek1 - start_time_cvek1 
cat("Computation time CVEK:", computation_time_cvek1, "seconds")

fit_CVEK1$beta #Fixed effect estimate of x on y.
fit_CVEK1$lambda #The selected tuning parameter based on the estimated ensemble kernel matrix.
fit_CVEK1$u_hat #A vector of weights of the kernels in the library.

#Testing the model 
pred_CVEK1 <- predict(fit_CVEK1, newdata = test)

plot(test$xobs, test$yobs, main = "Actual Observations vs. Predictons", 
     xlab = "x", ylab = "y", col = "blue", pch = 19)
points(test$xobs, pred_CVEK1, col = "red", type = "b")
legend(-2, 0.3, legend=c("Actual", "Predictions"), col = c("blue", "red"), lty=1:2, cex=0.8)

######################################################################################################

#GPFDA fit
start_time_gpr1 <- Sys.time()
fit_gpr1 <- GPFDA::gpr(response=as.matrix(train$yobs), input = as.matrix(train$xobs), Cov = "matern", gamma = 2, trace = 4)
end_time_gpr1 <- Sys.time()
computation_time_gpr1 <- end_time_gpr1 - start_time_gpr1
cat("Computation time GPR:", computation_time_gpr1, "seconds")

sapply(fit_gpr1$hyper, exp) #estimated parameters in log scale
fit_gpr1$var.hyper #estimation variance

summary(fit_gpr1)
plot(fit_gpr1, main = "Actual Observations vs. Predictons")

#Testing the model 
pred_gpr1 <- GPFDA::gprPredict(fit_gpr1, inputNew = test[, c(2)])
vignette("gpr_ex1", package = "GPFDA")
plot(pred_gpr1,main = "Actual Observations vs. Predictons")


######################################################################################################

#kernlab fit

set.seed(123)
start_time_gausspr1 <- Sys.time()
fit_gausspr1 <- gausspr(train$xobs, train$yobs, kernel = "rbfdot", variance.model = TRUE) 
end_time_gausspr1 <- Sys.time()
computation_time_gausspr1 <- end_time_gausspr1 - start_time_gausspr1
cat("Computation time GAUSSPR:", computation_time_gausspr1, "seconds")

fit_gausspr1

#Posterior Distribution of the model 
plot(train$xobs,train$yobs, main = "Actual Observations vs. Predictons (The Model)", 
     xlab = "x", ylab = "y", col = "blue", pch = 19)
xtest <- seq(min(train$xobs),max(train$xobs),0.2)
lines(xtest, predict(fit_gausspr1, xtest) , col = "red", type = "b") 
lines(xtest,
      predict(fit_gausspr1, xtest)+2*predict(fit_gausspr1,xtest, type="sdeviation"),
      col="gray")
lines(xtest,
      predict(fit_gausspr1, xtest)-2*predict(fit_gausspr1,xtest, type="sdeviation"),
      col="gray")
legend(-2, 0.3, legend=c("Actual", "Predictions", "CI95"), col = c("blue", "red", "gray"), lty=1:2, cex=0.8)

#Posterior Distribution for the test data. 
plot(test$xobs, test$yobs, main = "Actual Observations vs. Predictons (Test Data)", 
     xlab = "x", ylab = "y", col = "blue", pch = 19)
lines(test$xobs, predict(fit_gausspr1, newdata = test$xobs), col = "red", type = "b")
lines(test$xobs, #95CI upper bound
      predict(fit_gausspr1, newdata = test$xobs) + 
        2*predict(fit_gausspr1, newdata = test$xobs, type = "sdeviation"), 
      col = "gray")
lines(test$xobs,  #95CI lower bound
      predict(fit_gausspr1, newdata = test$xobs) - 
        2*predict(fit_gausspr1, newdata = test$xobs, type = "sdeviation"), 
      col = "gray")
legend(-2, 0.3, legend=c("Actual", "Predictions", "CI95"), col = c("blue", "red", "gray"), lty=1:2, cex=0.8)

#We can also calculate the mean and the covariance matrix by using \sigma estimation to calculate the CI95 manually as following;
x_predict <- seq(-5,5,len=50)

#Radial Basis Kernel 
SE <- function(Xi,Xj, l=1) exp(-0.5 * (Xi - Xj) ^ 2 / l ^ 2)
cov <- function(X, Y) outer(X, Y, SE)

#Mean and Covariance Matrix
sigma <- fit_gausspr1@kernelf@kpar[[1]] #sigma estimated by the model 
cov_xx_inv <- solve(cov(train$xobs, train$xobs) + sigma^2 * diag(1, length(train$xobs)))
Ef <- cov(x_predict, train$xobs) %*% cov_xx_inv %*% train$yobs
Cf <- cov(x_predict, x_predict) - cov(x_predict, train$xobs)  %*% cov_xx_inv %*% cov(train$xobs, x_predict)


#Posterior Distribution
require(ggplot2)
dat <- data.frame(x=x_predict, y=(Ef), ymin=(Ef-2*sqrt(diag(Cf))), ymax=(Ef+2*sqrt(diag(Cf))))    
ggplot(dat) +
  geom_ribbon(aes(x=x,y=y, ymin=ymin, ymax=ymax), fill="grey80") + # Var
  geom_line(aes(x=x,y=y), size=1) + #MEAN
  geom_point(data=train,aes(x=xobs,y=yobs)) +  #OBSERVED DATA
  scale_y_continuous(lim=c(-3,3), name="output, f(x)") + xlab("input, x")

######################################################################################################


#GauPro fit 

set.seed(123)
start_time_gaupro1 <- Sys.time()
fit_gp1 <- GauPro::GauPro(train$xobs, train$yobs, parallel=FALSE)
end_time_gaupro1 <- Sys.time()
computation_time_gaupro1 <- end_time_gaupro1 - start_time_gaupro1
cat("Computation time GauPro:", computation_time_gaupro1, "seconds")

#Distributions for the predictions
if (requireNamespace("MASS", quietly = TRUE)) {
  plot(fit_gp1)
}

fit_gp1$param.est
#Testing the model 

x <- test$xobs
y <- test$yobs

plot(x, y, main = "Actual Observations vs. Predictons (Test Data)", 
     xlab = "x", ylab = "y", col = "blue", pch = 19, ylim = c(-.4, 0.35))
curve(fit_gp1$predict(x), add=T, col=2)
curve(fit_gp1$predict(x)+2*fit_gp1$predict(x, se=T)$se, add=T, col="gray") #95% CI for upper bound
curve(fit_gp1$predict(x)-2*fit_gp1$predict(x, se=T)$se, add=T, col="gray") #95% CI for lower bound 
legend(-2, 0.35, legend=c("Actual", "Predictions", "CI95"), col = c("blue", "red", "gray"), lty=1:2, cex=0.8)

######################################################################################################


#REAL DATA EXAMPLE 

#Just Non-linear Interaction
#Simonsohn Data
setwd("~/Desktop")
data <- read.csv("dataa.csv")

#Model Fits 


glm 
start_time_glm <- Sys.time()
fit_glm <- glm(share~conservatism*c, data = data, family='binomial')
end_time_glm <- Sys.time()
computation_time_glm <- end_time_glm - start_time_glm
cat("Computation time GLM:", computation_time_glm, "seconds")

#gam 
start_time_gam <- Sys.time()
fit_gam <- mgcv::gam(share~te(conservatism,c),data = data, family='binomial')
end_time_gam <- Sys.time()
computation_time_gam <- end_time_gam - start_time_gam
cat("Computation time GAM:", computation_time_gam, "seconds")

#brms
start_time_brms <- Sys.time()
fit_brms <- brms::brm(share ~ gp(conservatism*c), data, chains = 2, family = bernoulli(), seed = 123) #family as bernoulli for binomial model
end_time_brms <- Sys.time()
computation_time_brms <- end_time_brms - start_time_brms
cat("Computation time BRMS:", computation_time_brms, "seconds")

summary(fit_brms)


#Predictions

#Low vs high moderator
sdk=1
xpredict=1:7
z.low  <- mean(data$c) - sd(data$c)*sdk
z.high <- mean(data$c) + sd(data$c)*sdk

#probit (it's actually logit, legacy comments and variable names)
probit.y.high <-predict(fit_glm , newdata = data.frame(c=z.high,conservatism=xpredict),type='response')
probit.y.low  <-predict(fit_glm , newdata = data.frame(c=z.low,conservatism=xpredict), type='response')


#gam.Spotlight  
gam.y.high <-predict(fit_gam , newdata = data.frame(c=z.high, conservatism=xpredict),type='response')
gam.y.low  <-predict(fit_gam , newdata = data.frame(c=z.low,conservatism=xpredict), type='response')


#brms prediction
brms.y.high <- predict(fit_brms, newdata = data.frame(c=z.high,conservatism=xpredict),type='response')
brms.y.low <- predict(fit_brms, newdata = data.frame(c=z.low,conservatism=xpredict),type='response')


#Unique participant id to bootstrap by it
data <- subset(data, non_US_ip==0)


# Function for quantiles to run through apply
q025 <- function(x) quantile(x,.025)
q975 <- function(x) quantile(x,.975)

#Automatically name elements in list with name of the objects in the list
namedList <- function(...) {
  L <- list(...)
  snm <- sapply(substitute(list(...)),deparse)[-1]
  if (is.null(nm <- names(L))) nm <- snm
  if (any(nonames <- nm=="")) nm[nonames] <- snm[nonames]
  setNames(L,nm)
}

# BOTH (gam and lm) simple slopes for the bootstrap study
both.simple.slope <- function(x,z,y,sdk=1,xpredict=1:7)
{
  #Estimate gam model
  # g <- gam(y~s(x,k=4)+s(z,k=4)+ti(x,z),family='binomial')
  g <- mgcv::gam(y~te(x,z),family='binomial')
  m <- glm(y~x*z,family='binomial')
  
  #Low vs high moderator
  z.low  <- mean(z) - sd(z)*sdk
  z.high <- mean(z) + sd(z)*sdk
  
  #gam.Spotlight  
  gam.y.high <-predict(g , newdata = data.frame(z=z.high,x=xpredict),type='response')
  gam.y.low  <-predict(g , newdata = data.frame(z=z.low,x=xpredict), type='response')
  
  #probit (it's actually logit, legacy comments and variable names)
  probit.y.high <-predict(m , newdata = data.frame(z=z.high,x=xpredict),type='response')
  probit.y.low  <-predict(m , newdata = data.frame(z=z.low,x=xpredict), type='response')
  
  
  return(namedList(gam.y.high , gam.y.low, probit.y.high, probit.y.low))
}


#BOOTSTRAP STUDY

set.seed(50)

#Unique participant id to bootstrap by it
participant.all <- unique(data$participant)

#Smaller dataset with just the 3 variables of interest
data.subset <- subset(data,select=c("share", "participant","c","conservatism"))

#How many bootstraps
btot=250

#Empty tables with results from bootstrap
gam.y.high.boot = gam.y.low.boot = 
  probit.y.high.boot = probit.y.low.boot = 
  matrix(nrow=btot,ncol=7)                  #empty matrix

#Bootstrap loop
for (bk in 1:btot){
  
  #Initiate empty dataframe
  data.boot <-data.frame(share=c(),c=c(),conservatism=c())
  
  #Draw participants IDs for this bootstrap
  participant.boot <- sample(participant.all,replace=TRUE)
  
  #Get their data
  for (k in 1:length(participant.boot)) {
    data.boot <-rbind(data.boot, subset(data.subset , participant==participant.boot[k]))
  }
  
  #
  s.boot <- both.simple.slope(x=data.boot$conservatism, z=data.boot$c, y=data.boot$share)
  gam.y.high.boot[bk,] <- s.boot$gam.y.high
  gam.y.low.boot[bk,] <- s.boot$gam.y.low
  probit.y.high.boot[bk,] <- s.boot$probit.y.high
  probit.y.low.boot[bk,] <- s.boot$probit.y.low
  cat("...",bk)
}


bootstrap.results<-namedList(gam.y.high.boot,    gam.y.low.boot, 
                             probit.y.high.boot, probit.y.low.boot)



#Compute 2.5th and 97.5th percentils
# GAM
yh1.g.lb <- apply(bootstrap.results$gam.y.high.boot,2,q025)
yh1.g.ub <- apply(bootstrap.results$gam.y.high.boot,2,q975)
yh0.g.lb <- apply(bootstrap.results$gam.y.low.boot,2,q025)
yh0.g.ub <- apply(bootstrap.results$gam.y.low.boot,2,q975)

#Logit
yh1.m.lb <- apply(bootstrap.results$probit.y.high.boot,2,q025)
yh1.m.ub <- apply(bootstrap.results$probit.y.high.boot,2,q975)
yh0.m.lb <- apply(bootstrap.results$probit.y.low.boot,2,q025)
yh0.m.ub <- apply(bootstrap.results$probit.y.low.boot,2,q975)

#Combine into single vector
#GAM CI
ci1.g <- c(yh1.g.ub, rev(yh1.g.lb))  #Confidence interval for high moderator
ci0.g <- c(yh0.g.ub, rev(yh0.g.lb))  #Confidence interval for low  moderator


#LOGIT  CI
ci1.m <- c(yh1.m.ub, rev(yh1.m.lb))
ci0.m <- c(yh0.m.ub, rev(yh0.m.lb))


# Margins
par(mfrow=c(1,2))

#Linear simple slopes
#LINES
plot  (1:7, probit.y.high,type='l',col='red',ylim=c(0,1),lty=2,lwd=2,ylab='',las=1,yaxt='n',xaxt='n',xlab='')
points(1:7 ,probit.y.low,type='l',col='blue',lwd=2)

#band
polygon(c(1:7,7:1),ci1.m, col=adjustcolor('red',.05), border=NA)
polygon(c(1:7,7:1),ci0.m, col=adjustcolor('blue',.05), border=NA)

#LETTER
text(1.15,.95,cex=3,"A")

#X-axis
axis(side=1,at=c(1:7))
axis(side=1,at=c(1,4,7),line=1,c("very liberal","neither", "very conservative"),tick=FALSE)
mtext(side=1,line=3.6,cex=1.5,font=2,"Political Orientation")

#y=axis
yt=seq(10,100,10)
axis(side=2,at=yt/100,paste0(yt,"%"),las=1)
mtext(side=2,line=3.5,cex=1.35,"Probability Sharing a Fake Story",font=2)
#Header
mtext(side=3,font=2,line=1.75,cex=1.35,'Linear Simple Slopes')

#legend
col2=adjustcolor('blue',.1)
col3=adjustcolor('red',.1)
legend('top',col=c('blue','red',col2,col3),
       lty=c(1,2,1,1),lwd=c(2,2,10,10), inset=.01, 
       cex=.80,
       legend=c("Low conscientiousness (-1 SD)",
                "High conscientiousness (+1 SD)",
                "95% Confidence band (boostrapped)",
                "95% Confidence band (boostrapped)"))




#GAM simple slopes
#LINES
plot(1:7, gam.y.high,type='l',col='red',ylim=c(0,1),lty=2,lwd=2,ylab='',las=1,yaxt='n',xaxt='n',xlab='')
points(1:7, gam.y.low,type='l',col='blue',lwd=2)

#Conf band
polygon(c(1:7,7:1),ci1.g, col=adjustcolor('red',.05), border=NA)
polygon(c(1:7,7:1),ci0.g, col=adjustcolor('blue',.05), border=NA)


#X-axis
axis(side=1,at=c(1:7))
axis(side=1,at=c(1,4,7),line=1,c("very liberal","neither", "very conservative"),tick=FALSE)
mtext(side=1,line=3.6,cex=1.5,font=2,"Political Orientation")

#y=axis
yt=seq(10,100,10)
axis(side=2,at=yt/100,paste0(yt,"%"),las=1)

#Header
mtext(side=3,font=2,line=1.75,cex=1.35,'GAM Simple Slopes')

#legend
col2=adjustcolor('blue',.1)
col3=adjustcolor('red',.1)
legend('top',col=c('blue','red',col2,col3),
       cex=.80,
       lty=c(1,2,1,1),lwd=c(2,2,10,10), inset=.01, 
       legend=c("Low conscientiousness (-1 SD)",
                "High conscientiousness (+1 SD)",
                "95% Confidence band (boostrapped)",
                "95% Confidence band (boostrapped)"))




#GP simple slopes
#LINES
plot(1:7, brms.y.high[,1],type='l',ylim = c(0, 1),col='red',lty=2,lwd=2,ylab='',las=1,yaxt='n',xaxt='n',xlab='')
points(1:7, brms.y.low[,1],type='l',col='blue',lwd=2)

#Conf band
polygon(c(1:7,7:1),as.vector(brms.y.high[, c(3,4)]), col=adjustcolor('red',.05), border=NA)
polygon(c(1:7,7:1),as.vector(brms.y.low[, c(3,4)]), col=adjustcolor('blue',.05), border=NA)

#X-axis
axis(side=1,at=c(1:7))
axis(side=1,at=c(1,4,7),line=1,c("very liberal","neither", "very conservative"),tick=FALSE)
mtext(side=1,line=3.6,cex=1.5,font=2,"Political Orientation")

#y=axis
yt=seq(10,100,10)
axis(side=2,at=yt/100,paste0(yt,"%"),las=1)
mtext(side=2,line=3.5,cex=1.35,"Probability Sharing a Fake Story",font=2)

#Header
mtext(side=3,font=2,line=1.75,cex=1.35,'GP Simple Slopes')

#legend
col2=adjustcolor('blue',.1)
col3=adjustcolor('red',.1)
legend("topleft",col=c('blue','red',col2,col3),
       cex=.80,
       lty=c(1,2,1,1),lwd=c(2,2,10,10), inset=.01, 
       legend=c("Low conscientiousness (-1 SD)",
                "High conscientiousness (+1 SD)",
                "95% Confidence band",
                "95% Confidence band"))

dev.off()
