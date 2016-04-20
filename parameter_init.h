// parameter_init.h
// function: initialize the parameter space (randomly, or read from some prior knowledge from GTEx for cis- coefficients)

#ifndef PARAMETER_INIT_H
#define PARAMETER_INIT_H

#include "global.h"

using namespace std;


// initializing the parameter space
void para_init();


// release all the dynamically allocated memory at the end of the program
void para_release();


// loading and preparing some gene (cis- relevant) mate data
void gene_cis_index_init();


// use the prior beta we get from GTEx to initialize the cis- regularization items
void beta_prior_fill();



#endif

// end of parameter_init.h