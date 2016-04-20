// opt_debugger.h
// function: debugging the optimization routine

#ifndef OPT_DEBUGGER_H
#define OPT_DEBUGGER_H


#include <string>
#include "lib_matrix.h"


using namespace std;




int para_check_nan(string etissue);


void para_temp_save_dev(int);


void para_temp_save_matrix_imcomp(Matrix_imcomp, char *);


void para_temp_save_matrix(Matrix, char *);


void para_temp_save_var(float *, int, char *);





#endif

// end of opt_debugger.h
