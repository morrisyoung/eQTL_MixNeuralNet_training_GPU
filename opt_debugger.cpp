// some debugger routines for the "optimization" routine (including all the involved subroutines)

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//#include <string.h>
#include <unordered_map>
#include <string>
#include <vector>
//#include "basic.h"
//#include <forward_list>
//#include <utility>
//#include "genotype.h"
//#include "expression.h"
//#include "optimization.h"
#include "global.h"
#include "main.h"  // typedef struct tuple_long
#include <math.h>       /* exp */
#include "opt_debugger.h"
#include "optimization.h"
#include "lib_matrix.h"




using namespace std;




// func:
//		0. this is for one specific tissue
//		1. check whether there are nan for any of the parameter (the actual parameter space, not the parameter dev space); if there is, return 1, otherwise return 0
//		2. also return where is the nan
int para_check_nan(string etissue)
{
	int etissue_index = etissue_index_map[etissue];

	int flag = 0;	// indicating whether or not there is any Nan n the parameter space

	//================================== vector<Matrix_imcomp> cube_para_cis_gene ===================================
	if(cube_para_cis_gene[etissue_index].check_nan())
	{
		cout << "I find Nan in para_cis_gene ..." << endl;
		flag = 1;
	}

	//================================== Matrix matrix_para_snp_cellenv ===================================
	if(matrix_para_snp_cellenv.check_nan())
	{
		cout << "I find Nan in para_snp_cellenv ..." << endl;
		flag = 1;
	}

	//============================== vector<Matrix> cube_para_cellenv_gene ==============================
	if(cube_para_cellenv_gene[etissue_index].check_nan())
	{
		cout << "I find Nan in para_cellenv_gene ..." << endl;
		flag = 1;
	}

	//=============================== Matrix matrix_para_batch_batch_hidden ===============================
	if(matrix_para_batch_batch_hidden.check_nan())
	{
		cout << "I find Nan in para_batch_batch_hidden ..." << endl;
		flag = 1;
	}

	//=============================== Matrix matrix_para_batch_hidden_gene ================================
	if(matrix_para_batch_hidden_gene.check_nan())
	{
		cout << "I find Nan in para_batch_hidden_gene ..." << endl;
		flag = 1;
	}


	return flag;
}




// func:
//		save all the parameter dev (five parts, for one specific tissue) into "../result_tempdata/"
void para_temp_save_dev(int etissue_index)
{
	//================================== vector<Matrix_imcomp> cube_para_dev_cis_gene ===================================
	char filename[100] = "../result_tempdata/para_dev_cis_gene.txt";
	para_temp_save_matrix_imcomp(cube_para_dev_cis_gene[etissue_index], filename);


	//================================== Matrix matrix_para_dev_snp_cellenv ===================================
	sprintf(filename, "%s", "../result_tempdata/para_dev_snp_cellenv.txt");
	para_temp_save_matrix(matrix_para_dev_snp_cellenv, filename);


	//============================== vector<Matrix> cube_para_dev_cellenv_gene[] ==============================
	sprintf(filename, "%s", "../result_tempdata/para_dev_cellenv_gene.txt");
	para_temp_save_matrix(cube_para_dev_cellenv_gene[etissue_index], filename);


	//=============================== Matrix matrix_para_dev_batch_batch_hidden ===============================
	sprintf(filename, "%s", "../result_tempdata/para_dev_batch_batch_hidden.txt");
	para_temp_save_matrix(matrix_para_dev_batch_batch_hidden, filename);


	//=============================== Matrix matrix_para_dev_batch_hidden_gene ================================
	sprintf(filename, "%s", "../result_tempdata/para_dev_batch_hidden_gene.txt");
	para_temp_save_matrix(matrix_para_dev_batch_hidden_gene, filename);

	return;
}



// func: save a Matrix_imcomp class into the file
void para_temp_save_matrix_imcomp(Matrix_imcomp matrix_imcomp, char * filename)
{
	long int dimension1 = matrix_imcomp.get_dimension1();

	FILE * file_out = fopen(filename, "w+");
	if(file_out == NULL)
	{
	    fputs("File error\n", stderr); exit(1);
	}

	for(long int i=0; i<dimension1; i++)
	{
		long int dimension2 = matrix_imcomp.get_dimension2(i);
		for(long int j=0; j<dimension2; j++)
		{
			float value = matrix_imcomp.get(i, j);
			char buf[1024];
			sprintf(buf, "%f\t", value);
			fwrite(buf, sizeof(char), strlen(buf), file_out);
		}
		fwrite("\n", sizeof(char), 1, file_out);
	}
	fclose(file_out);

	return;
}



// func: save a Matrix class (actually an matrix) into the file
void para_temp_save_matrix(Matrix matrix, char * filename)
{
	long int dimension1 = matrix.get_dimension1();
	long int dimension2 = matrix.get_dimension2();

	FILE * file_out = fopen(filename, "w+");
	if(file_out == NULL)
	{
	    fputs("File error\n", stderr); exit(1);
	}

	for(int i=0; i<dimension1; i++)
	{
		for(long j=0; j<dimension2; j++)
		{
			float value = matrix.get(i, j);
			char buf[1024];
			sprintf(buf, "%f\t", value);
			fwrite(buf, sizeof(char), strlen(buf), file_out);
		}
		fwrite("\n", sizeof(char), 1, file_out);
	}
	fclose(file_out);

	return;
}



// func: save a list of values from float *
void para_temp_save_var(float * list, int length, char * filename)
{
    FILE * file_out = fopen(filename, "w+");
    if(file_out == NULL)
    {
        fputs("File error\n", stderr); exit(1);
    }
	for(int i=0; i<length; i++)
	{
		float parameter = list[i];
		char buf[1024];
		sprintf(buf, "%f\n", parameter);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
	}
	fclose(file_out);
	//
	return;
}


