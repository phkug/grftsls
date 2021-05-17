# Two-stage least squares random forests

An extension of the [grf-package] to estimate heterogenous effects of instrumental variable models with multiple instruments using [two-stage least squares random forests].

## Installation
The package can be installed from source using ```devtools```:
```
devtools::install_github("phkug/grftsls", subdir = "r-package/grftsls")
```
Note that a compiler that implements C++11 is required (clang 3.3 or higher, or g++ 4.8 or higher). If installing on Windows, the [RTools toolchain] is also required.

If you want to make changes to the source code, we refer to the original documation of the [grf development guide]. 

## Using Example
```
library("grftsls")
library("grf")     # Note: The grf package is nessecary for centering.

n <- 2000
p <- 10
k <- 5
X <- matrix(rnorm(n * p, 1, 0.5), n, p)
Z <- matrix(rnorm(n * k, 1, 0.5), n, k)
Q <- matrix(rbinom(n * k, 1, 0.5), n, k)
W <- data.matrix(rowSums(Q * Z))

tau <-  X[, 1] / 2
Y <- data.matrix(rowSums(X[, 1:3]) + tau * W + rowSums(Q) + rnorm(n))

# Build a Forest (incl. tuning and centering). 
tsls.forest <- tsls_forest(X, Y, W, Z, Y.hat = NULL, W.hat = NULL, Z.hat = NULL) 
# Note: Centering of Y,W,Z is not conducted by default.  

# Predict on out-of-bag training samples.
tsls.pred <- predict(tsls.forest, estimate.variance = T)
```

## Literature
[Biewen, M., & Kugler, P. (2021). Two-stage least squares random forests with an application to Angrist and Evans (1998). Economics Letters, 109893.]

[Biewen, M., Kugler, P., 2020. Two-stage least squares random forests with an application to Angrist and Evans (1998). In: IZA Discussion Paper, (13613). Institute for the Study of Labor (IZA), Bonn.]

[Athey, S., Tibshirani, J., Wager, S., 2019. Generalized random forests. Ann. Statist. 47, 1148–1178.]




[grf-package]: https://github.com/grf-labs/grf
[two-stage least squares random forests]: https://www.sciencedirect.com/science/article/abs/pii/S0165176521001701
[RTools toolchain]: https://cran.r-project.org/bin/windows/Rtools
[grf development guide]: https://grf-labs.github.io/grf/DEVELOPING.html

[Biewen, M., Kugler, P., 2020. Two-stage least squares random forests with an application to Angrist and Evans (1998). In: IZA Discussion Paper, (13613). Institute for the Study of Labor (IZA), Bonn.]: https://www.iza.org/publications/dp/13613/two-stage-least-squares-random-forests-with-an-application-to-angrist-and-evans-1998
[Biewen, M., & Kugler, P. (2021). Two-stage least squares random forests with an application to Angrist and Evans (1998). Economics Letters, 109893.]: https://www.sciencedirect.com/science/article/abs/pii/S0165176521001701
[Athey, S., Tibshirani, J., Wager, S., 2019. Generalized random forests. Ann. Statist. 47, 1148–1178.]: https://projecteuclid.org/journals/annals-of-statistics/volume-47/issue-2/Generalized-random-forests/10.1214/18-AOS1709.short



