#include <R.h>
#include <Rmath.h>
#include <Rinternals.h>


typedef struct logZ_linked_list_t_s logZ_linked_list_t;

struct logZ_linked_list_t_s {
  double part;
  logZ_linked_list_t *next;
};

typedef struct {
  logZ_linked_list_t *list;
  double max;
  int size;
  double log_tol_min;
  double log_tol_step;
  double log_tol_max;
} logZ_register_t;


#define ADD_LOG(a, b) { if (a==R_NegInf) { a = b; } else {  if (a>b) { a = a+log1p(exp(-a+b)); } else { a = b+log1p(exp(-b+a)); }}}


inline logZ_linked_list_t* alloc_logZ_linked_list(double part, logZ_linked_list_t* next) {
  logZ_linked_list_t* new=(logZ_linked_list_t*)R_alloc(1, sizeof(logZ_linked_list_t));
      new->part=part;
      new->next=next;
      return new;
}


void merge_logZ_linked_list(logZ_linked_list_t* list, double log_tol_step, double* new_max) {
  if (list!=NULL) {
    if (list->next!=NULL) {
      if (list->part>log_tol_step+log(list->next->part)+list->next->part) {
	list->part=list->next->part+log1p(exp(list->part-list->next->part));
	list->next=list->next->next;
	if (list->next==NULL) {
	  *new_max=list->part;
	} else {
	  merge_logZ_linked_list(list, log_tol_step, new_max);
	}
      } 
    } else {
      *new_max=list->part;
    }
  }
}


  
logZ_linked_list_t* add_logZ_linked_list(logZ_linked_list_t* list, double part, double log_tol_min, double log_tol_step,  int* stop, double* new_max) {
  logZ_linked_list_t* cur=list;
  //  logZ_linked_list_t* prev;
  // List is empty

  if (list==NULL) {
    *new_max=part;
    return alloc_logZ_linked_list(part, NULL);  
  }
  // We might need to insert at the very front
  if (list->part>part) {
    if (part>log_tol_step+log(list->part)+list->part) {
      list->part+=log1p(exp(part-list->part));
      merge_logZ_linked_list(list, log_tol_step, new_max);
      return list;
    } else {
      if (part<log_tol_min+log(list->part)+list->part)
	*stop=1;
      return alloc_logZ_linked_list(part, list);
    }    
  }
  while (1) {
    // We might need to insert before the next element or add
    if ((cur->next==NULL) || (cur->next->part>part)) {
      if ((cur->part>log_tol_step+log(part)+part) && ((cur->next==NULL) || (cur->next->part-part>part-cur->part))) {
	cur->part=part+log1p(exp(cur->part-part));
	merge_logZ_linked_list(cur, log_tol_step, new_max);
	// CHECK WHETHER WE NEED TO MERGE WITH NEXT ONE.
	return list;
      }
      if ((cur->next==NULL) || (part<log_tol_step+log(cur->next->part)+cur->next->part)) {
	if (cur->next==NULL)
	  *new_max=part;
	cur->next=alloc_logZ_linked_list(part, cur->next);
	return list;
      }
      cur->next->part+=log1p(exp(part-cur->next->part));
      merge_logZ_linked_list(cur->next, log_tol_step, new_max);
      // CHECK WHETHER WE NEED TO MERGE WITH NEXT ONE.
      return list;
    }
    cur=cur->next;
  }
  Rprintf("We should never end up here");
}



void print_logZ_linked_list(logZ_linked_list_t* list) {
  if (list==NULL)
    return;
  Rprintf("x <- c(x,%.30e)\n",list->part);
  print_logZ_linked_list(list->next);

}

R_INLINE int add_logZ_component(logZ_register_t* reg, double part) {
  double max=R_NegInf;
  int stop=0;
  reg->size++;
  reg->list=add_logZ_linked_list(reg->list, part,  reg->log_tol_min, reg->log_tol_step, &stop, &max);
  if (max>R_NegInf) {
    reg->max=max;
  } 
  if (part<reg->log_tol_max+log(reg->max)+reg->max) 
      stop=2;
  return stop;
}

double sum_logZ_register(logZ_register_t* reg, int* count) {
  logZ_linked_list_t* cur=reg->list;
  *count=0;
  if (cur==NULL) return R_NegInf;
  double sum=cur->part;
  *count=1;
  while ((cur=cur->next)!=NULL) {
    sum=cur->part+log1p(exp(sum-cur->part));
    (*count)++;
  }
  return sum;
}



logZ_register_t* alloc_register(double log_tol_min, double log_tol_max, double log_tol_step) {
  logZ_register_t* reg=(logZ_register_t*)R_alloc(1,sizeof(logZ_register_t));
  reg->list=NULL;
  reg->log_tol_min=log_tol_min;
  reg->log_tol_max=log_tol_max;
  reg->log_tol_step=log_tol_step;
  reg->size=0;
  reg->max=R_NegInf;
  return reg;
}

#define ADD_LOG_CHECK(a, b, temp, prec) { if (a==R_NegInf) { a = b; } else { if (a>b) { temp=exp(-a+b); if (temp<prec*fabs(a)) break; a = a+log1p(temp); } else { temp=1; a = b+log1p(exp(-b+a)); } } }

R_INLINE int addup(double *logZ, double cur, logZ_register_t* reg, double sub_prec) {
  double new_term=exp(-(*logZ)+cur);
  int result=0;
  if (new_term<sub_prec*fabs(*logZ)) {
    result=add_logZ_component(reg, *logZ);
    *logZ=cur;
  } else { 
    *logZ = *logZ + log1p(new_term);
  }
  return result;
}
 
R_INLINE int addup_check(double *logZ, double cur,  logZ_register_t* reg, double sub_prec) {
  int result=0;
  double new_term;
  if (*logZ<cur) {
    *logZ = cur + log1p(exp(*logZ-cur));
  } else {
    new_term=exp(-(*logZ)+cur);
    if (new_term<sub_prec*fabs(*logZ)) {
      result=add_logZ_component(reg, *logZ);
      //      if (result!=0) Rprintf("RESULT = %i\n",result);
      *logZ=cur;
    } else { 
      *logZ = *logZ + log1p(new_term);
    }
  }
  return result;
}
 

// Parameters log_pmf_prec

double comlogSumtransform(double mu, double nu, double normaliser, double start, int up, double log_pmf_prec, double sub_prec, int max_iter, int reset_cycle, logZ_register_t* reg, double* end) {
  double i, cur, logZ/*, initial_cur*/;
  int count;
  int reason=0;
  i=start;
  count=reset_cycle;
  cur=nu*(i*log(mu)-lgammafn(i+1))-normaliser;
  //  cur=i*log_lambda-nu*lgammafn(i+1)-normaliser;
  logZ=cur;
  if (up || (i-1!=i))
    while (1) {  
      if (cur<log_pmf_prec) { reason=3; break; }
      if (up) {
	if (((max_iter>0) && (i>start+max_iter)) || (i+1==i)) {
	  reason=4;
	  break;
	}
	i++;
      } else {
	if ((i<1) || ((max_iter>0) && (i<start-max_iter))) {	
	  reason=4;
	  break;
	}
      i--;
      }
      R_CheckUserInterrupt();
      count--;
      if (count==0) {
	count=reset_cycle;
	cur=nu*(i*log(mu)-lgammafn(i+1))-normaliser;
	//	cur=i*log_lambda-nu*lgammafn(i+1)-normaliser;
      } else {
	if (up) {
	  cur=cur+nu*(log(mu)-log(i));

 // cur=cur+log_lambda-nu*log(i);
	} else {
	  cur=cur-nu*(log(mu)-log(i+1));
  //  cur=cur-log_lambda+nu*log(i+1);
	}
      }
      if ((reason=addup(&logZ, cur, reg, sub_prec))!=0)
	break;
    }
  add_logZ_component(reg, logZ);
  *end=i;
  return R_NegInf; 
}

#define GEO_SERIES(log_r, k) log1p(-exp((log_r)*(k)))-log1p(-exp(log_r))

double logzcmpois(double mu, double nu, double from, double to, double current_logZ, int tails, double log_pmf_prec, double log_tail_prec, double log_tol_min, double log_tol_max, double log_tol_step, double tol_add, int max_iter, int max_tail_iter, int reset_cycle, double initial_step_size, double step_multiplier, double* new_from, double* new_to, double* log_tails_lower, double* log_tails_upper) {
  double  mode, term, normaliser;
  char * r_vmax = vmaxget();
  logZ_register_t* reg=alloc_register(log_tol_min, log_tol_max, log_tol_step);
   int size;
   //  mu=exp(log_lambda/nu);
  mode=floor(mu);
  normaliser=nu*(mode*log(mu)-lgammafn(mode+1));
 // normaliser=mode*log_lambda-nu*lgammafn(mode+1);
  if (current_logZ==R_NegInf) {
    from=mode; 
    to=mode-1;
  } else {
    add_logZ_component(reg, current_logZ-normaliser);
  }
  // SUM UP PROBABILITIES FROM LEFT HALF (STARTING AT MODE) 
  if (from>0) {
    term=comlogSumtransform(mu, nu, normaliser, from-1, 0, log_pmf_prec, tol_add, max_iter, reset_cycle, reg, &from);
  }
  // SUM UP PROBABILITIES FROM RIGHT HALF (STARTING AT MODE) 
  term=comlogSumtransform(mu, nu, normaliser, to+1, 1, log_pmf_prec, tol_add, max_iter, reset_cycle, reg, &to);
  current_logZ=sum_logZ_register(reg,&size);
  //  Rprintf("REGISTER SIZE : %i / %i\n",size, reg->size);

  /*  print_logZ_linked_list(reg->list);

  Rprintf("MAX = %e\n",reg->max);
  */ 

  vmaxset(r_vmax);
  *new_from=from;
  *new_to=to;
  if (tails) {
    logZ_register_t* lower=alloc_register(log_tol_min, log_tol_max, log_tol_step);
    logZ_register_t* upper=alloc_register(log_tol_min, log_tol_max, log_tol_step);
    // TAIL BOUNDS (LOWER TAILS)
    double cur,next_step, lower_logZ, upper_logZ, i, log_r, step_size;
    int j;
    lower_logZ=R_NegInf;
    upper_logZ=R_NegInf;
    if (from>0) {
      step_size=initial_step_size;
      i=from-1;
      j=0;
      while (i-step_size==i) 
	step_size=round(step_size*step_multiplier);	
      if (step_size>i) step_size=i+1;
      while (i>0) {
	R_CheckUserInterrupt();
	next_step=i-step_size;
	if ((j++>max_tail_iter) || (next_step<-1)) {
	  next_step=-1;
	  step_size=i;
	}
	//      Rprintf("Working on (%f,%f] : ",next_step,i, step_size, initial_step_size, step_multiplier);
	cur=nu*(i*log(mu)-lgammafn(i+1))-normaliser;
//cur=i*log_lambda-nu*lgammafn(i+1)-normaliser;
	// lower bound
        log_r=-nu*(log(mu)-log(next_step+2));
	//log_r=-log_lambda+nu*log(next_step+2);
	//	Rprintf("cur = %e / log_r = %e\n", cur,log_r);
	if (log_r==0)
	  term=cur+log(step_size);
	else
	  term=cur+GEO_SERIES(log_r, step_size);
	addup_check(&lower_logZ, term, lower, tol_add);
	// upper bound
	log_r=-nu*(log(mu)-log(i));
        //log_r=-log_lambda+nu*log(i);
	if (log_r==0)
	  term=cur+log(step_size);
	else
	term=cur+GEO_SERIES(log_r, step_size);
	//	Rprintf("(L) upper_logZ = %e / term=%e\n",upper_logZ,term);
	if ((addup_check(&upper_logZ, term, upper, tol_add)!=0) || (term<log_tail_prec))
	  break;
	step_size=round(step_size*step_multiplier);
	i=next_step;
      }
      if (i>0) {
	// Do the last block
	cur=nu*(i*log(mu)-lgammafn(i+1))-normaliser;
        //cur=i*log_lambda-nu*lgammafn(i+1)-normaliser;
	// lower bound
	log_r=-nu*log(mu);
        //log_r=-log_lambda;
	if (log_r==0)
	  term=cur+log(i);
	else
	  term=cur+GEO_SERIES(log_r, i);
	addup_check(&lower_logZ, term, lower, tol_add);
	// upper_bound
	log_r=-nu*(log(mu)-log(i));
        //log_r=-log_lambda+nu*log(i);
	if (log_r==0)
	  term=cur+log(i);
	else
	  term=cur+GEO_SERIES(log_r, i);
	addup_check(&upper_logZ, term, upper, tol_add);
      }
      add_logZ_component(lower, lower_logZ);
      add_logZ_component(upper, upper_logZ);
    }
    // TAIL BOUNDS (UPPER TAILS)
    lower_logZ=R_NegInf;
    upper_logZ=R_NegInf;
    step_size=initial_step_size;
    i=to+1;
    while (i-step_size==i)
      step_size=round(step_size*step_multiplier);	
    j=0;
    while (1) {
      R_CheckUserInterrupt();
      next_step=i+step_size;
      if ((next_step==i) || (j++>max_tail_iter))
	break;
      //      Rprintf("Working on (%f,%f] : ",next_step,i, step_size, initial_step_size, step_multiplier);
      cur=nu*(i*log(mu)-lgammafn(i+1))-normaliser;
      //cur=i*log_lambda-nu*lgammafn(i+1)-normaliser;
      if (!R_FINITE(cur))
	break;
      // lower bound
      log_r=nu*(log(mu)-log(next_step-1));
      //log_r=log_lambda-nu*log(next_step-1);
      if (log_r==0)
	term=cur+log(step_size);
      else
	term=cur+GEO_SERIES(log_r, step_size);
      addup_check(&lower_logZ, term, lower, tol_add);
      // upper bound
      log_r=nu*(log(mu)-log(i+1));
      //log_r=log_lambda-nu*log(i+1);
      if (log_r==0)
	term=cur+log(step_size);
      else
      term=cur+GEO_SERIES(log_r, step_size);
      //      Rprintf("(R) upper_logZ = %e / term=%e / cur=%e /  log_r=%e (%i)\n",upper_logZ,term,cur,log_r,log_r==0);
      if ((addup_check(&upper_logZ, term, upper, tol_add)!=0) || (term<log_tail_prec)) 
	break;
      step_size=round(step_size*step_multiplier);
      i=next_step;
    }
    //    Rprintf("i = %f / next_step = %f\n",i,next_step);
    // Finish with the tail ...
    cur=nu*(next_step*log(mu)-lgammafn(next_step+1))-normaliser;
    //cur=next_step*log_lambda-nu*lgammafn(next_step+1)-normaliser;  // next_step -> i
    if (R_FINITE(cur)) {
      log_r=nu*(log(mu)-log(next_step+1));
      // log_r=log_lambda-nu*log(next_step+1); // next_step -> i
      term=cur-log1p(-exp(log_r));
      addup_check(&upper_logZ, term, upper, tol_add);
    }
    add_logZ_component(lower, lower_logZ);
    add_logZ_component(upper, upper_logZ);
    // Add everything up ...
    add_logZ_component(lower, current_logZ);
    add_logZ_component(upper, current_logZ);
    *log_tails_lower=sum_logZ_register(lower,&size)+normaliser;
    *log_tails_upper=sum_logZ_register(upper,&size)+normaliser;
    /*
    Rprintf("max = %e\n",upper->max);
    print_logZ_linked_list(upper->list);
    Rprintf("REGISTER SIZE (UPPER TAIL) : %i / %i\n",size, upper->size);
    */
    vmaxset(r_vmax);
  } else {
    *log_tails_lower=R_NegInf;
    *log_tails_upper=R_PosInf;
  }
  return current_logZ+normaliser;
}





void logzcmpois_R(double* mu, double* nu, int* n, double* from, double* to, double* current_logZ, int* tails, double* tol_pmf, double* tol_tails, double* tol_min, double* tol_max, double* tol_step, double* tol_add, int* max_iter, int* max_tail_iter, int* reset_cycle, double* initial_step_size, double* step_multiplier, double* new_logZ, double* new_from, double* new_to,  double* log_tails_lower, double* log_tails_upper) {
  int i;
  for (i=0; i<*n; i++) {
    new_logZ[i]=logzcmpois(mu[i], nu[i], *from, *to, *current_logZ, *tails, log(*tol_pmf), log(*tol_tails), log(*tol_min), log(*tol_max), log(*tol_step), *tol_add, *max_iter, *max_tail_iter, *reset_cycle, *initial_step_size, *step_multiplier, &new_from[i], &new_to[i], &log_tails_lower[i], &log_tails_upper[i]);
  }
}

// Efficient simulation from the COM Poisson distribution 

