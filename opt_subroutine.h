// opt_subroutine.h
// function: subroutines of optimization

#ifndef OPT_SUBROUTINE_H
#define OPT_SUBROUTINE_H


#include <string>
#include "global.h"
#include "lib_matrix.h"
#include <array>
#include <vector>


using namespace std;



// activation function for the neural network
void neuralnet_ac_func(float *, int);


float neuralnet_ac_func_dev(float);


// transform the sample ID (like "GTEX-R55E-0826-SM-2TC5M") into individual ID (here is the first 9 digits)
string sample_to_individual(string);


// the main optimization routine: the forward_backward propagation, and the gradient descent
void forward_backward_prop_batch(string, int, int);
void forward_backward(int, array<float *, NUM_CHR> *, vector<float> *, float *, float *, float *, float *, Matrix_imcomp, Matrix, Matrix, Matrix, Matrix);
void regularization(int);
void gradient_descent(int);


// calculate the log-likelihood for one tissue
float cal_loglike(string);


// forward process, and return the log-likelihood
float forward_loglike(string, int);

float forward_loglike_testerror(int, string, int, array<float *, NUM_CHR> *, float *, float *, float *, float *);



// calculate the test errors for one tissue
float cal_testerror(string);




#endif

// end of opt_subroutine.h
