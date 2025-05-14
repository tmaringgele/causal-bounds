# Load package
library(causaloptim)
library(igraph)
set.seed(42)

# 1. Simulate data with Z -> X -> Y and confounding from U to both X and Y
n <- 5000
Z <- rbinom(n, 1, 0.5)   # Instrument
U <- rbinom(n, 1, 0.5)   # Confounder (treated as latent in model)

# Generate X from Z and U
prob_X <- plogis(-1 + 1.5 * Z + 1.2 * U)
X <- rbinom(n, 1, prob_X)

# Generate Y from X and U with fixed causal effect of X
beta_X <- 1.0
beta_U <- 1.5
prob_Y <- plogis(-0.5 + beta_X * X + beta_U * U)
Y <- rbinom(n, 1, prob_Y)

# Create dataframe and marginalize out U
df <- data.frame(Z = Z, X = X, Y = Y)
marginals <- prop.table(table(df$Y, df$X, df$Z), margin = 3)

# Extract probabilities in the p_yx_z format
p00_0 <- marginals["0", "0", "0"]
p10_0 <- marginals["1", "0", "0"]
p01_0 <- marginals["0", "1", "0"]
p11_0 <- marginals["1", "1", "0"]

p00_1 <- marginals["0", "0", "1"]
p10_1 <- marginals["1", "0", "1"]
p01_1 <- marginals["0", "1", "1"]
p11_1 <- marginals["1", "1", "1"]

# 2. Define the causal graph with Ur as latent
graph <- graph_from_literal(Z -+ X, X -+ Y, Ur -+ X, Ur -+ Y)
V(graph)$leftside <- c(1, 0, 0, 0)
V(graph)$latent   <- c(0, 0, 0, 1)
V(graph)$nvals    <- c(2, 2, 2, 2)
E(graph)$rlconnect     <- c(0, 0, 0, 0)
E(graph)$edge.monotone <- c(0, 0, 0, 0)

# 3. Specify the PNS query and compute bounds
pns_query <- "p{Y(X = 1) = 1; Y(X = 0) = 0}"
obj <- analyze_graph(graph, constraints = NULL, effectt = pns_query)
bounds <- optimize_effect_2(obj)
boundsfunc <- interpret_bounds(bounds = bounds$bounds, obj$parameters)

# 4. Evaluate bounds with simulated data
PNS_bounds <- boundsfunc(
  p00_0 = p00_0, p00_1 = p00_1,
  p10_0 = p10_0, p10_1 = p10_1,
  p01_0 = p01_0, p01_1 = p01_1,
  p11_0 = p11_0, p11_1 = p11_1
)

# 5. Calculate true PNS from counterfactual outcomes
logit_Y1 <- -0.5 + beta_X * 1 + beta_U * U  # If X=1
logit_Y0 <- -0.5 + beta_X * 0 + beta_U * U  # If X=0
pY1 <- 1 / (1 + exp(-logit_Y1))
pY0 <- 1 / (1 + exp(-logit_Y0))

Y1 <- rbinom(n, 1, pY1)
Y0 <- rbinom(n, 1, pY0)

PNS_true <- mean(Y1 == 1 & Y0 == 0)

# Output
PNS_bounds
PNS_true

# Clean up
closeAllConnections()
rm(list=ls())
