# R-Packages-for-GP
Available R-Packages for Gaussian Process Regression
```

Gaussian Process (GP) regression and Generalized Additive Models (GAMs) are both popular methods used in statistical modeling and machine learning for regression tasks. This work aims to compare available R packages to fit GP regression with each other and also with GAM. In the first part, the following R functions will be compared in terms of computation time, quantification of degree of non-linearity, posterior uncertainty and testing multiple predictors (i.e., multiple $X_{i}$);

 - `gam` from `"mgcv"`
 - `brm` from `"brms"`
 * `cvek` from `"CVEK"`
 * `gpr` from `"GPFDA"`
 * `gausspr` from `"kernlab"`
 * `GauPro` from `"GauPro"`
 

**The more detailed information regarding each package can be found in the package documentations, this work is only focused on the GP regression function in each package. **

In the second part, a real data example will be presented to compare GP regression with GAM and GLM by testing the hypothesis "the sharing of fake news is largely driven by low conscientiousness conservatives" from the study by Lawson and Kakkar (2021). 
 
 \

### summary 

You can see the summary of the properties that the packages have in below: 
```{r}
sum <- matrix(c(rep(NA, 24)), ncol = 6, nrow = 4)
colnames(sum) <- c("gam", "brm", "cvek", "gpr", "gausspr", "GauPro")
rownames(sum) <- c("degre of non-linearity", "posterior uncertainity", "testing", "testing multiple predictors")
sum[, 1] <- c("+", "+", "+", "+")
sum[, 2] <- c("+", "+", "+", "+")
sum[, 3] <- c("+", "-", "+", "+")
sum[, 4] <- c("+", "+", "-", "-")
sum[, 5] <- c("+", "+", "-", "-")
sum[, 6] <- c("-", "+", "-", "-")


knitr::kable(sum)

```

### Generalized Additive Models (GAMs)
GAMs are a type of regression model that allow for flexible, non-linear relationships between the predictor variables and the response variable. GAMs accomplish this by modeling the response variable as a sum of smooth functions of the predictor variables, rather than a linear combination. This allows for more complex and nuanced relationships to be captured, which can be particularly useful when the true relationship between variables is not well understood or linear.

A generalized additive model for a response variable **$Y$**, modeled as a function of predictor variables **$X_{1}, X_{2}, ..., X_{n}$** can be expressed as:

$$
g(E(Y)) = \beta_{0} + f_{1}(X_{1}) + f_{2}(X_{2}) + ... + f_{p}(X_{n}) + \epsilon
$$
 where **$f_{i}$** are smooth, non-linear functions of the predictor variables **$X_{i}$**, and **$\epsilon$** is a random error term. The functions **$f_{i}$** can be modeled using a variety of smoothing methods, such as splines, kernel methods, or generalized cross-validation.

\qquad 

### Gaussian Process  (GP) Regression 
On the other hand, Gaussian Process regression is a probabilistic regression model that is based on the assumption that the response variable follows a Gaussian distribution. GP regression can be seen as a generalization of linear regression, as it allows for non-linear relationships between the predictor variables and the response variable as well. However, rather than modeling the response variable as a sum of smooth functions of the predictor variables, GP regression models the response variable as a distribution over functions. This allows for greater flexibility in modeling complex relationships and also provides a measure of uncertainty in the predictions. The regression function modeled by a multivariate Guassian can be expressed as: 

$$
P(f|X) = N(f|\mu, K)
$$


where $X=[x_{1}, ..., x_{n}], f =[f(x_{1}), ..., f(x_{n})], \mu = [m(x_{1}, ..., m(x_{n}))]$, and $K_{ij} = k(x_{i}, x_{j})$. $X$ are observed data points, $m$ represents the mean function, and $k$ represent a positive definite kernel function, which is Radial Basis Function Kernel in this work. Notice that with no observation, the mean function is default to be $m(X) = 0$ given that the data is often normalized to a zero mean. The GP model is a distribution over functions and $K$ defines the shape (smoothness) of this distribution. In particular, if the points $x_{i}$ and $x_{j}$ are considered similar by the kernel, then the function outputs of them, $f(x_{i})$ and $f(x_{j})$ are expected to be similar. RBF kernel can be expressed as: 

$$
k(x_{i}, x_{j}) = exp(- \frac{||x_{i}, x_{j}||^2} {2\l^2})
$$
where $l$ is a free positive parameter, also called $length$ parameter. Higher values of $l$ leads to smoother curves whereas lower values leads to more wiggly curves. 
\qquad

One of the key differences between GAMs and GP regression is their underlying assumptions about the nature of the relationships between variables. GAMs assume that the relationships are smooth and can be represented by a sum of smooth functions, while GP regression assumes that the relationships are continuous and can be represented by a distribution over functions. Additionally, GAMs are typically simpler to implement and interpret, while GP regression can be computationally intensive and requires more expertise to use effectively. Ultimately, the choice between GAMs and GP regression will depend on the specific nature of the data and the modeling task at hand.

