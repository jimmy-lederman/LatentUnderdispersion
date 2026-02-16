#' Evaluate the logarithm of the normalisation constant of the COM-Poisson distribution.
#'
#' This function evaluates the logarithm of the normalisation constant of the
#' COM-Poisson distribution using geometric bounds. The implementation aims to
#' give as accurate results as possible and gives more reliable results than a
#' naive truncated sum of the p.m.f.
#' 
#' The parametrisation used is for \eqn{x\in\{0,1,2,\ldots\}}
#' \deqn{P\{X=x\}\propto\frac{\lambda^x}{(x!)^{\nu}}=\left(\frac{\mu^x}{x!}\right)^{\nu}}
#' i.e. \eqn{\lambda=\mu^{\nu}}.
#' 
#' The normalisation constant is the \eqn{Z(\lambda,\nu)} required such that
#' \eqn{\frac{1}{Z(\lambda,\nu)}\cdot \frac{\lambda^x}{(x!)^{\nu}}} is a valid
#' probability mass function, i.e. it sums to one.
#'
#' @param lambda Parameter \eqn{\lambda} of the COM-Poisson distribution. Can be a vector.
#' @param mu Parameter \eqn{\mu} of the COM-Poisson distribution. Can be a vector. If both \eqn{\mu} and \eqn{\lambda} are given \eqn{\lambda} will be ignored.
#' @param nu Parameter \eqn{nu} of the COM-Poisson distribution. Must be of the same length as \code{lambda} or \code{mu}.
#' @param previous.result A vector of results obtained from calling the function previously. In this case the existing results are refined. The code does not check the validity of \code{previous.results}.
#' @param tails Whether tail bounds should be computer (default \code{TRUE}).
#' @param max.iter Maximum number of iterations (refinements). Setting it to \code{-1} (default) enables unlimited number of iterations.
#' @param tol.pmf Numerical tolerance for evaluating the probability mass function (default \code{.Machine$double.eps}).
#' @param control List with control parameters allowing fine-tuning of the algorithm. See the package source for details.
#' @return Returns the logarithm of the normalisation constant for each parameter value supplied. The returned vector has am attribute \code{"details"} which contains a matrix providing additional information (smallest \eqn{x} (column \code{"from"}) and largest \eqn{x} (column \code{"to"}) for which the p.m.f. was summed up exactly as well as value of the logairhtm of that exact sum (column \code{"exact"}) and resulting lower and upper bounds (columns \code{"lower"} and \code{"upper"})).
#' @export
#' @useDynLib combayes logzcmpois_R
#' @examples
#' logzcmpois(lambda=10, nu=1)
#' logzcmpois(lambda=10, nu=0.5)
#' logzcmpois(lambda=10, nu=2)

logzcmpois <- function(lambda, mu=lambda^(1/nu), nu, previous.result=NULL, tails=TRUE, max.iter=-1, tol.pmf=.Machine$double.eps, control=list()) {
  default.control <- list(initial.step.size=2, step.multiplier=2, tol.tails=.Machine$double.eps, tol.min=.Machine$double.eps, tol.max=.Machine$double.eps^1.25,
                          tol.step=.Machine$double.eps^0.75, tol.add=.Machine$double.eps^0.75, 

                          max.tail.iter=100,reset.cycle=100, strategy="midpoint")
  for (name in names(default.control)) {
    if (!(name %in% names(control)))
      control[[name]] <- default.control[[name]]
  }
  list.diff <- setdiff(names(control),names(default.control))
  if (length(list.diff)>0) {
    warning.text <- "The following arguments in the control list were ignored:"
    for (elt in list.diff)
      warning.text <- paste(warning.text," \"",elt,"\"",sep="")
    warning(warning.text)
  }
  pra <- attr(previous.result, "details")
  if (is.null(pra) || (ncol(pra)<3)) {
    pra <- cbind(NA, NA, rep(-Inf,length(mu)))
  }
  if (length(mu)!=length(nu))
    stop("mu and mu must be of the same length.")
  result <- .C(logzcmpois_R, mu=as.double(mu), nu=as.double(nu), n=as.integer(length(mu)), from=as.double(pra[,1]), to=as.double(pra[,2]), current_logZ=as.double(pra[,3]), tails=as.integer(tails), tol.pmf=as.double(tol.pmf), tol.tails=as.double(control$tol.tails), tol.min=as.double(control$tol.min), tol.max=as.double(control$tol.max), tol.step=as.double(control$tol.step), tol.add=as.double(control$tol.add), max_iter=as.integer(max.iter), max_tail_iter=as.integer(control$max.tail.iter), reset_cycle=as.integer(control$reset.cycle), initial_step_size=as.double(control$initial.step.size), step_multiplier=as.double(control$step.multiplier), new_logZ=double(length(mu)), new_from=double(length(mu)), new_to=double(length(mu)), log_tails_lower=double(length(mu)), log_tails_upper=double(length(mu)), NAOK=TRUE)
  strategies <- c("truncated","lower","upper","midpoint")
  control$strategy <- strategies[pmatch(control$strategy, strategies)]
  if (!tails)
    control$strategy <- "truncated"
  answer <- switch(control$strategy,
                  truncated=result$new_logZ,
		  lower=result$log_tails_lower,
		  upper=result$log_tails_upper,
		  midpoint=(result$log_tails_lower+result$log_tails_upper)/2)
  attr(answer,"details") <- cbind(from=result$new_from, to=result$new_to, exact.part=result$new_logZ, lower=result$log_tails_lower, upper=result$log_tails_upper)
  answer
}

#' Evaluate the density of the COM-Poisson distribution.
#'
#' This function evaluates the probability mass function of the COM-Poisson
#' distribution using a close to exact algorithm for calculating the
#' normalisation constant
#' 
#' The parametrisation used is for \eqn{x\in\{0,1,2,\ldots\}}
#' \deqn{P\{X=x\}\propto\frac{\lambda^x}{(x!)^{\nu}}=\left(\frac{\mu^x}{x!}\right)^{\nu}}
#' i.e. \eqn{\lambda=\mu^{\nu}}.
#' The normalisation constant is the \eqn{Z(\lambda,\nu)} required such that
#' \eqn{\frac{1}{Z(\lambda,\nu)}\cdot \frac{\lambda^x}{(x!)^{\nu}}} is a valid
#' probability mass function, i.e. it sums to one.
#'
#' @param x Value of \eqn{x} for which the p.m.f. is to be evaluated. Can be a vector.
#' @param lambda Parameter \eqn{\lambda} of the COM-Poisson distribution. Can be a vector, but must in that case be of the same length as \code{x}.
#' @param mu Parameter \eqn{\mu} of the COM-Poisson distribution. Can be a vector, but must in that case be of the same length as \code{x}. If both \eqn{\mu} and \eqn{\lambda} are given \eqn{\lambda} will be ignored.
#' @param nu Parameter \eqn{nu} of the COM-Poisson distribution. Must be of the same length as \code{lambda} or \code{mu}.
#' @param unnormalised Whether the unnormalised p.m.f. is to be returned (default \code{FALSE})
#'@param log Whether the logarithm of the p.m.f. is to be returned (default \code{FALSE})
#'@param ... Additional arguments passed on to \code{\link{logzcmpois}}
#' @return Vector of the p.m.f. as specified in the arguments.
#' @export
#' @examples
#' # Compre densities of COM-Poisson distribution with different nu
#' x <- 0:25
#' dcmpois(x, lambda=10, nu=1)
#' dcmpois(x, lambda=10, nu=0.5)
#' dcmpois(x, lambda=10, nu=2)
#' matplot(x, cbind(dcmpois(x, mu=10, nu=1),
#'                  dcmpois(x, mu=10, nu=0.5),
#'                  dcmpois(x, mu=10, nu=2)), type="o", col=2:4, pch=16, ylab="p.m.f.")
#' legend("topright", col=2:4, lty=1:3, c(expression(nu*"="*1),
#'                                        expression(nu*"="*0.5),
#'                                        expression(nu*"="*2)))

dcmpois <- function(x, lambda, mu=lambda^(1/nu), nu, unnormalised=FALSE, log=FALSE, ...) {
    mu <- as.double(mu)
    
    if (length(nu) == 1L)
      nu <- rep_len(nu, length(mu))
    
    nu <- as.double(nu)
    if (length(mu)!=length(nu))
        stop("mu and mu must be of the same length.")
    if (unnormalised) {
        norm <- 0
    } else {
        if (length(mu)==1) {
            norm <- logzcmpois(mu=mu, nu=nu, ...)
        } else {
            if (length(mu)!=length(x))
                stop("mu must be of length 1 or of the same length as x.")
            norm <- logzcmpois(mu=mu, nu=nu, ...)
        }
        attributes(norm) <- NULL
    }
    result <- nu*(ifelse(x==0, 0, x*log(mu) - lgamma(x+1))) - norm
    if (!log)
        result <- exp(result)
    result
}
