// opt_hierarchy.h
// function: MLE given hierarchy (for parental nodes)

#ifndef OPT_HIERARCHY_H
#define OPT_HIERARCHY_H


#include <string>
#include "lib_matrix.h"
#include <array>
#include <vector>


using namespace std;




void matrix_multiply(vector<vector<float>> *, vector<vector<float>> *, vector<float> *, vector<float> *);


void hierarchy_matrix_prepare(vector<vector<float>> *, vector<vector<float>> *);


void hierarchy();



#endif

// end of opt_hierarchy.h
