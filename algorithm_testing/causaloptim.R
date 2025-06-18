library(causaloptim)

graph <- igraph::graph_from_literal(Z -+ X, X -+ Y, 
                                    Ur -+ X, Ur -+ Y)

b <- igraph::graph_from_literal(X -+ Y, X -+ M, M -+ Y, 
                                Ur -+ Y, Ur -+ M)
V(b)
V(graph)
V(graph)$leftside <- c(1, 0, 0, 0)
V(graph)$latent   <- c(0, 0, 0, 1)
V(graph)$nvals    <- c(2, 2, 2, 2)
E(graph)$rlconnect     <- c(0, 0, 0, 0)
E(graph)$edge.monotone <- c(0, 0, 0, 0)

riskdiff <- "p{Y(X = 1) = 1} - p{Y(X = 0) = 1}"
obj <- analyze_graph(graph, constraints = NULL, effectt = riskdiff)
bounds <- optimize_effect_2(obj)
boundsfunction <- interpret_bounds(bounds = bounds$bounds, obj$parameters)

#pyx_z represents P(Y=y, X=x | Z = z)
# Joint distribution P(Y, X | Z) for binary variables

# P(Y=0, X=0 | Z=0)
p00_0 <- 0.35
# P(Y=0, X=0 | Z=1)
p00_1 <- 0.20

# P(Y=1, X=0 | Z=0)
p10_0 <- 0.25
# P(Y=1, X=0 | Z=1)
p10_1 <- 0.10

# P(Y=0, X=1 | Z=0)
p01_0 <- 0.15
# P(Y=0, X=1 | Z=1)
p01_1 <- 0.30

# P(Y=1, X=1 | Z=0)
p11_0 <- 0.25
# P(Y=1, X=1 | Z=1)
p11_1 <- 0.40

boundsfunction(p00_0 = p00_0, p00_1 = p00_1, 
               p10_0 = p10_0, p10_1 = p10_1, 
               p01_0 = p01_0, p01_1 = p01_1, 
               p11_0 = p11_0, p11_1 = p11_1)

help("causaloptim")
