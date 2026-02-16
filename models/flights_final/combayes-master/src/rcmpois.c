#include <R.h>
#include <Rmath.h>
#include <float.h>

// Evaluate unnormalised density of the COM-poisson distribution 
// if fudge is set to lgammafn(mode+1) then the unnormalised density is one at the mode
// If mode and fudge are set to 0 then the usual unnormalised density is computed
double unnorm_ldcpois(double x, double mu, double nu, double mode, double fudge) {
  return nu*((x-mode)*log(mu)-lgammafn(x+1)+fudge);
}

// Sample from a geometric distribution truncated to {0,1,...,n}
// u is a U[0,1] realisation
double R_INLINE truncated_geo_sample(double u, double logq, double n) {
  double C;
  if (logq>-DBL_EPSILON)
    return 0;
  C = -expm1(logq*(n+1));
  return floor(log(1-C*u)/logq);
}

// Sample from a geometric distribution with range {0,1,2,...}
// u is a U[0,1] realisation
double R_INLINE untruncated_geo_sample(double u, double logq) {
  if (logq>-DBL_EPSILON)
    return 0;
  return floor(log(u)/logq);
}

// Compute the finite geometric series (1+q+...+q^deltax)
double R_INLINE truncated_lweights(double deltax, double logq) {
  if (logq>-DBL_EPSILON)
    return log(deltax+1)+logq;
  return log1p(-exp((deltax+1)*logq)) - log1p(-exp(logq));
}

// Compute the geometric series (1+q+...)
double R_INLINE untruncated_lweights(double logq) {
  return -log1p(-exp(logq));
}
  

int rcmpois(double mu, double nu, int n, double* result) {
  double logmu, lmode, rmode, fudge, sd, lsd, rsd, maxlweight, logprob, x, u;
  int i, attempts;
  double ldens[4];
  double lweights[4];
  double logq[4];
  double sweights[4];
  logmu = log(mu);
  // Figure out mode and standard deviation
  lmode = ceil(mu)-1;
  fudge = lgammafn(lmode+1);
  rmode = lmode+1;
  fudge = lgammafn(lmode+1);
  sd = ceil(sqrt(mu)/sqrt(nu));
  if (sd<5)
    sd = 5;
  // Set up two points at mode +/- sd
  lsd = round(lmode-sd);
  if (lsd<0)
    lsd=-1;
  rsd = round(rmode+sd);
  // Left most tail
  if (lsd==-1) {
    lweights[0] = R_NegInf;
    logq[0] = 0;
    ldens[0] = R_NegInf;
  } else {
    ldens[0] = unnorm_ldcpois(lsd, mu, nu, lmode, fudge);
    if (lsd==0) {
      lweights[0] = ldens[0];
      logq[0] = 0;
    } else {
      logq[0] = nu * (-logmu + log(lsd));    
      lweights[0] = ldens[0] + truncated_lweights(lsd, logq[0]);
    }
  }
  // within 1sd to the left of the mode
  ldens[1] = 0;
  if (lmode==0) {
    lweights[1] = 0;
    logq[1] = 1;
  } else {
    logq[1] = nu * (-logmu + log(lmode));
    lweights[1] = truncated_lweights(lmode-lsd-1, logq[1]);
  }
  // within 1sd to the right of the mode
  logq[2] = nu * (logmu - log(rmode+1));
  ldens[2] = nu * (logmu - log(rmode));
  lweights[2] = ldens[2] + truncated_lweights(rsd-rmode-1, logq[2]);
  // right tail
  logq[3] = nu * (logmu - log(rsd+1));
  ldens[3] = unnorm_ldcpois(rsd, mu, nu, lmode, fudge);
  lweights[3] = ldens[3] + untruncated_lweights(logq[3]);

  // Find maximum log-weight
  maxlweight=lweights[0];
  for (i=1; i<4; i++)
    if (lweights[i]>maxlweight) maxlweight=lweights[i];

  // Compute the cumulative sum of the weights  
  for (i=0; i<4; i++)
    lweights[i]=lweights[i]-maxlweight;
  sweights[0]=exp(lweights[0]);
  for (i=1; i<4; i++)
    sweights[i]=sweights[i-1]+exp(lweights[i]);

  // Draw the sample by rejection sampling
  attempts=0;
  for (i=0; i<n; i++) {
    while (1) {
      attempts++;
      u = unif_rand() * sweights[3];
      if (u < sweights[0]) {  
	u = u / sweights[0];
	x = truncated_geo_sample(u, logq[0], lsd);
	logprob=ldens[0]+x*logq[0];
	x = lsd-x;
      } else {
	if (u<sweights[1]) {
	  u = (u-sweights[0])/(sweights[1]-sweights[0]);
	  x = truncated_geo_sample(u, logq[1], lmode-lsd-1);
	  logprob =ldens[1]+x*logq[1];
	  x = lmode - x;
	} else {
	  if (u<sweights[2]) {
	    u = (u-sweights[1])/(sweights[2]-sweights[1]);
	    x = truncated_geo_sample(u, logq[2], rsd-rmode-1);
	    logprob =ldens[2]+x*logq[2];
	    x = rmode + x;
	  } else {
	    u = (u-sweights[2])/(sweights[3]-sweights[2]);
	    x = untruncated_geo_sample(u, logq[3]);
	    logprob =ldens[3]+x*logq[3];
	    x = rsd + x;
	  }
	}
      }
      if (log(unif_rand())<unnorm_ldcpois(x, mu, nu, lmode, fudge) - logprob) {
	result[i]=x;
	break;
      }
    }
  }
  return attempts;
}

// classical R/C interface
void rcmpois_wrapper(double* mu, double* nu, int* npars, int* nout, double* result, int* attempts) {
  int i;
  GetRNGstate();
  for (i=0; i<*npars; i++) {
    attempts[i]=rcmpois(mu[i], nu[i], *nout, &result[i*(*nout)]);
  }
  PutRNGstate();
}


