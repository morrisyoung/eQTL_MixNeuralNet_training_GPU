// libfunc_matrix.h

#ifndef LIBFUNC_MATRIX_H
#define LIBFUNC_MATRIX_H


#include <iostream>
//#include <sys/types.h>
//#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <vector>
#include <unordered_map>
#include "lib_matrix.h"
#include <array>
#include "global.h"




using namespace std;




//============================================ task specific functions =================================================
void para_penalty_lasso_approx(Matrix, Matrix, float, float);

void para_penalty_cis(Matrix_imcomp, Matrix_imcomp, vector<vector<float>>, float, float, float);


void para_gradient_descent(Matrix, Matrix, float);

void para_gradient_descent_cis(Matrix_imcomp, Matrix_imcomp, float);


//============================================ abstract functions =================================================
void multi_array_matrix(float *, Matrix, float *);

void multi_array_matrix_imcomp(array<float *, NUM_CHR> *, Matrix_imcomp, float *);

void multi_array_list_matrix(array<float *, NUM_CHR> *, Matrix, float *);


//============================================ back propogation =================================================
void backward_error_prop_direct_imcomp(Matrix_imcomp, float *, array<float *, NUM_CHR> *);

void backward_error_prop_last_layer(Matrix, float *, float *);

void backward_error_prop_inter_layer_1(float *, Matrix, Matrix, float *, array<float *, NUM_CHR> *);

void backward_error_prop_inter_layer_2(float *, Matrix, Matrix, float *, float *);




#endif

// end of libfunc_matrix.h

