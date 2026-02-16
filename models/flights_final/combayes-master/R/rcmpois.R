#' Draw samples from the COM-Poisson distribution.
#'
#' This function draws samples from the COM-Poisson distribution using an exact
#' rejection sampling approach using piece-wise geometric rejection envelope.
#' The parametrisation used is for \eqn{x\in\{0,1,2,\ldots\}}
#' \deqn{P\{X=x\}\propto\frac{\lambda^x}{(x!)^{\nu}}=\left(\frac{\mu^x}{x!}\right)^{\nu}}
#' i.e. \eqn{\lambda=\mu^{\nu}}.
#' 
#' @param lambda Parameter \eqn{\lambda} of the COM-Poisson distribution. Can be a vector.
#' @param mu Parameter \eqn{\mu} of the COM-Poisson distribution. Can be a vector. If both \eqn{\mu} and \eqn{\lambda} are given \eqn{\lambda} will be ignored.
#' @param nu Parameter \eqn{nu} of the COM-Poisson distribution. Must be of the same length as \code{lambda} or \code{mu}.
#'
#' @param n Number of samples drawn per parameter values provided (default 1).
#' @return Samples from the COM Poisson distribution with the given parameters (\code{n} each).
#' @export
#' @useDynLib combayes rcmpois_wrapper
rcmpois <- function(n = 1, lambda, mu = lambda^(1/nu), nu) {
  
  mu <- as.double(mu)
  
  if (length(nu) == 1L)
    nu <- rep_len(nu, length(mu))
  
  nu <- as.double(nu)
  
  if (length(mu) != length(nu))
    stop("mu and nu must have same length or nu must be scalar.")
  
  result <- .C(rcmpois_wrapper,
               mu = mu,
               nu = nu,
               n = as.integer(length(mu)),
               nout = as.integer(n),
               result = double(n * length(mu)),
               attempts = integer(length(mu)),
               DUP = FALSE)
  
  if (length(mu) > 1 && n > 1)
    result$result <- matrix(result$result, ncol = n, byrow = TRUE)
  
  attr(result$result, "accept") <- n / result$attempts
  result$result
}
