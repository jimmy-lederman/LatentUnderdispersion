#' Bayesian regression for COM-Poisson distributed data

#' This function performs Bayesian inference in a COM-Poisson regression model of the form \deqn{\mu_i=\exp(x_i'\beta)\mbox{ and }\nu_i=\exp(-x_i'\delta)}, which corresponds to \deqn{E(Y_i)\approx\exp(x_i'\beta)\mbox{ and }\mathrm{Var}(Y_i)=\exp(x_i'(\beta+\delta)).}
#' 
#' @param y Vector of observations.
#' @param X Design matrix (without column of 1's for the intercept). Number of rows must be equal to the length of \code{y}.
#' @param num_samples Number of MCMC samples to be drawn (excluding burn-in).
#' @param burnin Number of MCMC samples drawn during the burn-in.
#' @param algorithm Algorithm to be used, either \code{"exchange"} or \code{"bound"}.
#' @param empirical_cov Empirical covariance of the posterior samples obtained from a previous run. Optional. If omitted the empirical covariance of the posterior samples drawn during burn-in is used instead.
#' @param  update_emp_cov_cycle If the empircial covariance of the ppsterior samples is calculated during burnin, every how many samples should this occur (default 1000).
#' @param update_scaling Multiplicative factor for for proposal standard deviation (default 2).
#' @param initial_beta Initial value of \eqn{\beta}. Calculated from a frequentist quasi-Poisson GLM if omitted.
#' @param initial_delta Initial value of \eqn{\beta}. Initialised automatically if omitted.
#' @param prior_mean_beta Prior mean of \eqn{\beta} (default to a zero vector).
#' @param prior_var_beta Prior covariance of \eqn{\beta} (default to the identity matrix times 100).
#' @param prior_mean_delta Prior mean of \eqn{\beta} (default to a zero vector).
#' @param prior_var_delta Prior covariance of \eqn{\beta} (default to the identity matrix times 100).
#' @param random_seed Random seed to be used for the samling. If omitted the current state of the RNG is used.
#' @param ... Additional arguments passed to the sampler (used by the bounds sampler to control how the bounds are calculated.
#' @return A list with the following entries
#' \item{posterior_beta}{Draws from the posterior distribution of \eqn{\beta}}
#' \item{posterior_delta}{Draws from the posterior distribution of \eqn{\delta}}
#' \item{posterior_beta}{Draws from the posterior distribution of \eqn{\beta}}
#' \item{accept_beta}{Acceptance rate of the Gibbs updates of \eqn{\beta}}
#' \item{accept_delta}{Acceptance rate of the Gibbs updates of \eqn{\delta}}
#' \item{accept_mixed}{Acceptance rate of the mixed Gibbs updates the i-th components of \eqn{\beta} and \eqn{delta}.}
#' @examples
#' \dontrun{# Simulate toy data (from a Poisson distribution)
#' n <- 1000
#' x <- runif(n)
#' y <- rcmpois(mu=exp(1+x), nu=exp(-1.5+x))
#' # Fit MCMC
#' result <- cmpoisreg(y, x, num_samples=5e3, burnin=1e3)
#' colMeans(result$posterior_beta)
#' colMeans(result$posterior_delta)}
#' @export
#' 
cmpoisreg <- function(y, X, num_samples, burnin, algorithm="exchange", empirical_cov, update_emp_cov_cycle=1e3, update_scaling=2, initial_beta, initial_delta, prior_mean_beta, prior_var_beta, prior_mean_delta, prior_var_delta, random_seed, ...){

    conditional_var <- function(S, idx) {
        S[idx,idx]-S[idx,-idx]%*%solve(S[-idx,-idx])%*%S[-idx,idx]
    }

                                        #Accept/reject decision in case we decide to choose the exchange algorithm
	accept_move_exchange <- function(y,mu_new,nu_new,mu,nu,log_u,...){
	    ystar <- rcmpois(mu=mu_new,nu=nu_new,n=1) #Sample new data y* from proposal
		    log_likeli_ratio <- sum(dcmpois(y,mu=mu_new,nu=nu_new,log=TRUE,unnormalised=TRUE)-dcmpois(y,mu=mu,nu=nu,log=TRUE,unnormalised=TRUE))+sum(-dcmpois(ystar,mu=mu_new,nu=nu_new,log=TRUE,unnormalised=TRUE)+dcmpois(ystar,mu=mu,nu=nu,log=TRUE,unnormalised=TRUE))
	    list(accept=log_likeli_ratio>log_u,details=list())
	}

	#Accept/reject decision in case we decide to choose the bounds algorithm
	accept_move_bounds <- function(y,mu_new,nu_new,mu,nu,log_u,...){
	    control <- list(...)
	    log_rhs <- -log_u+sum(dcmpois(y,mu=mu_new,nu=nu_new,log=TRUE,unnormalised=TRUE)-dcmpois(y,mu=mu,nu=nu,log=TRUE,unnormalised=TRUE))
	    comlog_new<- NULL
	    comlog <- NULL
	    iterations <- control$iterations
	    while(TRUE){
	        comlog_new<- logzcmpois(mu=mu_new,nu=nu_new,#previous.result=comlog_new,
	                                      max.iter=iterations)
	        comlog <- logzcmpois(mu=mu,nu=nu,#previous.result=comlog,
	                                   max.iter=iterations)
	        low_new<- sum(attr(comlog_new,"details")[,"lower"])
	        high_new <- sum(attr(comlog_new,"details")[,"upper"])
	        low <- sum(attr(comlog,"details")[,"lower"])
	        high <- sum(attr(comlog,"details")[,"upper"])
	        if (low_new-high>log_rhs) {
	            accept <- FALSE
	            break
	        }
	        if (high_new-low<log_rhs) {
	            accept <- TRUE
	            break
	        }
	        # Now we need to refine
        	iterations <- iterations * control$iterations.multiplier
	    }
	    list(accept=accept, details=list())
	}

    accept_move <- c(accept_move_exchange, accept_move_bounds)[[pmatch(algorithm, c("exchange","bounds"))]]
        
    #Add intercept column on the design matrix and save the number of columns
    #set.seed(1)
    design_matrix <- as.matrix(cbind(1,X))
    p <- dim(design_matrix)[2]
    
                                        #Prior mean and variance for parameters beta and delta
    if (missing(prior_mean_beta))
        prior_mean_beta  <- rep(0,p)
    if (missing(prior_var_beta))
        prior_var_beta <-  diag(p)*100
    if (missing(prior_mean_delta))        
        prior_mean_delta <- 0
    if (missing(prior_var_delta))            
        prior_var_delta <-  100

    if (missing(initial_beta) || missing(initial_delta) || missing(empirical_cov))
        freq_glm <- stats::glm(y ~ design_matrix-1, family="quasipoisson")
    if (missing(initial_beta))
        initial_beta <- stats::coef(freq_glm)
    if (missing(initial_delta))
        initial_delta <- log(summary(freq_glm)$dispersion)
    update_empirical_cov <- FALSE
    if (missing(empirical_cov)) {
        empirical_cov <- matrix(c(1,0,0,1),ncol=2)%x%summary(freq_glm)$cov.scaled
        #empirical_cov <- diag(2*p) * 0.1
        update_empirical_cov <- TRUE        
    }

    generate_updates <- function(empirical_cov) {
        sel <- 1:p
        mixed <- list()
        for (i in sel)
            mixed[[i]] <- conditional_var(empirical_cov,i+c(0,p))*update_scaling^2
        list(beta=conditional_var(empirical_cov,1:p)*update_scaling^2,
             delta=update_scaling^2,
             mixed=mixed)
    }

    update_cov <- generate_updates(empirical_cov)
    
    if (!missing(random_seed))
        set.seed(random_seed)
    #Initial (and current) values for parameters beta and delta
    beta_current  <- drop(mvnfast::rmvn(1,mu=initial_beta,sigma=empirical_cov[1:p,1:p]))
    delta_current <- drop(mvnfast::rmvn(1,mu=initial_delta,sigma=1))
    
    #Initial (and current) values for mu and nu
    mu_current <- exp(design_matrix%*%beta_current)
    nu_current <- exp(-delta_current)
    
    #Progress bar to check the progress of the MCMC
    pb <- utils::txtProgressBar(min=1, max=num_samples, style=3)
   #Set values for the number of acceptances
    accept_beta  <- 0
    accept_delta <- 0
    accept_mixed <- 0
    #Define matrices to save the posterior draws into
    posterior_beta  <- matrix(0,nrow=num_samples+burnin,ncol=p)
    posterior_delta <- matrix(0,nrow=num_samples+burnin,ncol=1)
    
    for (l in 1:(num_samples+burnin)){#Start of the MCMC algorithm


# UPDATE empirical_cov if needed
        if ((l==burnin || (l<burnin && l%%update_emp_cov_cycle==0) )&& update_empirical_cov) {
            empirical_cov <- stats::var(cbind(posterior_beta,posterior_delta))
            update_cov <- generate_updates(empirical_cov)

        }

        
           for (i in 1:p) {
               par_candidate <- drop(mvnfast::rmvn(1,mu=c(beta_current[i],delta_current[i]), sigma=update_cov$mixed[[i]]))
               beta_candidate <- beta_current
               beta_candidate[i] <- par_candidate[1]
               delta_candidate <- delta_current
               delta_candidate[i] <- par_candidate[2]
               print(delta_current)
               print(prior_mean_delta)
               print(prior_var_delta)
               log_u <- log(stats::runif(1)) + mvnfast::dmvn(beta_current,mu=prior_mean_beta,sigma=prior_var_beta,log=TRUE)- mvnfast::dmvn(beta_candidate,mu=prior_mean_beta,sigma=prior_var_beta,log=TRUE)  + mvnfast::dmvn(delta_current,mu=prior_mean_delta,sigma=prior_var_delta,log=TRUE)- mvnfast::dmvn(delta_candidate,mu=prior_mean_delta,sigma=prior_var_delta,log=TRUE)             
               mu_candidate <- exp(design_matrix%*%beta_candidate)
               nu_candidate <-  exp(-delta_candidate)
               if(accept_move(y=y,mu_new=mu_candidate,nu_new=nu_candidate,mu=mu_current,nu=nu_current,log_u=log_u,...)$accept){
                   beta_current <- beta_candidate
                   mu_current   <- mu_candidate
                   delta_current <- delta_candidate
                   nu_current   <- nu_candidate
                   
                   #Keep number of acceptances also
                   accept_mixed <- accept_mixed+1/p
           }



           }


           
           #######################################
           #Updating multivariate parameter beta##
           #######################################
           #Sample two uniform distibuted variables and use one for each M-H or exchange step
           #Propose new parameter
           beta_candidate <- drop(mvnfast::rmvn(1,mu=beta_current,sigma=update_cov$beta))#This needs to change later on (empirical covariance matrix)
           mu_candidate <- exp(design_matrix%*%beta_candidate)
           nu_candidate <- nu_current
           #Estimate the log of (uniform * ratio of priors). This is what we defined as log_u within the decision algorithm
           log_u <- log(stats::runif(1)) +mvnfast::dmvn(beta_current,mu=prior_mean_beta,sigma=prior_var_beta,log=TRUE)-mvnfast::dmvn(beta_candidate,mu=prior_mean_beta,sigma=prior_var_beta,log=TRUE)
           #Decision_algorithm refers either to the exchange or the bounds
           if(accept_move(y=y,mu_new=mu_candidate,nu_new=nu_candidate,mu=mu_current,nu=nu_current,log_u=log_u,...)$accept){#Bounds algorithm: we have to add iterations and iterations.multiplier
               beta_current <- beta_candidate
               mu_current   <- mu_candidate
               #Keep number of acceptances also
               accept_beta <- accept_beta+1
           }
           ########################################
           #Updating multivariate parameter delta##
           ########################################
           #Propose new parameter
           delta_candidate <- drop(mvnfast::rmvn(1,mu=delta_current,sigma=update_cov$delta))#This needs to change later on (empirical covariance matrix)
           mu_candidate <- mu_current
           nu_candidate <- exp(-design_matrix%*%delta_candidate)
           #Estimate the log of (uniform * ratio of priors). This is what we defined as log_u within the decision algorithm
           log_u <- log(stats::runif(1)) +mvnfast::dmvn(delta_current,mu=prior_mean_delta,sigma=prior_var_delta,log=TRUE)-mvnfast::dmvn(delta_candidate,mu=prior_mean_delta,sigma=prior_var_delta,log=TRUE)
           #Decision_algorithm refers either to the exchange or the bounds
           if(accept_move(y=y,mu_new=mu_candidate,nu_new=nu_candidate,mu=mu_current,nu=nu_current,log_u=log_u,...)$accept){#Bounds algorithm: we have to add iterations and iterations.multiplier
              delta_current <- delta_candidate
              nu_current    <- nu_candidate 
              #Keep number of acceptances also
              accept_delta <- accept_delta+1
           }
       utils::setTxtProgressBar(pb, l)#Check the progress of the MCMC
      posterior_beta[l,] <- beta_current; posterior_delta[l,] <- delta_current #Save all draws after the burn-in period
    }#end of the MCMC
    close(pb)
    list(posterior_beta=posterior_beta[-(1:burnin),],posterior_delta=posterior_delta[-(1:burnin),],accept_beta=accept_beta/(burnin+num_samples),accept_delta=accept_delta/(burnin+num_samples),accept_mixed=accept_mixed/(burnin+num_samples))
}
