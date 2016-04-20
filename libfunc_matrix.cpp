// this contains the operations on the matrix type



#include "libfunc_matrix.h"
#include "lib_matrix.h"
#include <math.h>
#include <string>
#include <vector>
#include <unordered_map>
#include "global.h"
#include <array>
#include "opt_nn_acfunc.h"
#include <math.h>       /* exp */





using namespace std;





// Apr.1: let's do the multi-threading for regularization and gradient descent routines:
//	void para_penalty_lasso_approx(Matrix matrix_para, Matrix matrix_para_dev, float lambda, float sigma)
//	void para_penalty_cis(Matrix_imcomp matrix_imcomp_para, Matrix_imcomp matrix_imcomp_para_dev, vector<vector<float>> repo_prior, float lambda_lasso, float lambda_ridge, float sigma)
//	void para_gradient_descent(Matrix matrix_para, Matrix matrix_para_dev, float rate)
//	void para_gradient_descent_cis(Matrix_imcomp matrix_imcomp_para, Matrix_imcomp matrix_imcomp_para_dev, float rate)
// I can do this because this happens after the whole mini-batch, so there are no multi-threading any more
// I will write the multi-threading version of these routines in "libfunc_matrix_mt.cpp"




//============================================ task specific functions =================================================
// func: add the LASSO penalty term in the matrix_para_dev; destroy type
//		the matrix_para is used to calculate the derivative term, that will be added to the matrix_para_dev
void para_penalty_lasso_approx(Matrix matrix_para, Matrix matrix_para_dev, float lambda, float sigma)
{
	long int dimension1 = matrix_para.get_dimension1();
	long int dimension2 = matrix_para.get_dimension2();


	for(int i=0; i<dimension1; i++)
	{
		for(int j=0; j<dimension2; j++)
		{
			/// the value of current para beta:
			float beta = matrix_para.get(i, j);
			/// the derivative of the beta:
			float derivative = beta / sqrt (beta * beta + sigma);  // this is an approximation of the LASSO regularization
			/// and the value of its derivative should be added with that derivative term for regularization:
			matrix_para_dev.add_on(i, j, lambda * derivative);
		}
	}

	return;
}



// func:
//		regularization for the cis- parameters, with both lasso and ridge (composed as elastic net)
//		for sparsity and pruning signal
void para_penalty_cis(Matrix_imcomp matrix_imcomp_para, Matrix_imcomp matrix_imcomp_para_dev, vector<vector<float>> repo_prior, float lambda_lasso, float lambda_ridge, float sigma)
{

	long int dimension1 = matrix_imcomp_para.get_dimension1();
	for(long int i=0; i<dimension1; i++)
	{
		int chr = matrix_imcomp_para.get_chr(i);
		long int sst = matrix_imcomp_para.get_sst(i);

		long int dimension2 = matrix_imcomp_para.get_dimension2(i);
		for(long int j=0; j<dimension2; j++)
		{
			long int pos = j + sst;  // the pos in snp list

			/// the value of current cis- beta:
			float beta = matrix_imcomp_para.get(i, j);

			/// the prior that we need (if there is) for tuning the relative strength of L1 and L2 regularization:
			// TODO: we don't load this currently; NOTE that we don't have the prior information for the intercept term (the last term)
			//prior = prior_tissue_rep[etissue][chr-1][pos];
			float prior = 1.0;

			float alpha = 1 / ( 1 + exp(-(prior-1)) );

			/// the derivative of the beta:
			float derivative1 = beta / sqrt (beta * beta + sigma);  // this is an approximation of the LASSO regularization
			float derivative2 = 2 * beta;  // L2 regularization item is differentiable

			/// and the value of its derivative should be added with that derivative item from regularization:
			float value = lambda_lasso * (1 - alpha) * derivative1 + lambda_ridge * alpha * derivative2;
			matrix_imcomp_para_dev.add_on(i, j, value);
		}

	}

	return;
}



// func: do the gradient descent for this parameter matrix
//		with: beta = beta - rate * dev
void para_gradient_descent(Matrix matrix_para, Matrix matrix_para_dev, float rate)
{
	long int dimension1 = matrix_para.get_dimension1();
	long int dimension2 = matrix_para.get_dimension2();

	for(long int i=0; i<dimension1; i++)
	{
		for(long int j=0; j<dimension2; j++)
		{
			float dev = matrix_para_dev.get(i, j);
			matrix_para.add_on(i, j, - rate * dev);
		}
	}

	return;
}


// func: do the gradient descent for this parameter matrix_incomp
//		with: beta = beta - rate * dev
void para_gradient_descent_cis(Matrix_imcomp matrix_imcomp_para, Matrix_imcomp matrix_imcomp_para_dev, float rate)
{
	long int dimension1 = matrix_imcomp_para.get_dimension1();
	for(long int i=0; i<dimension1; i++)
	{
		long int dimension2 = matrix_imcomp_para.get_dimension2(i);
		for(long int j=0; j<dimension2; j++)
		{
			float dev = matrix_imcomp_para_dev.get(i, j);
			matrix_imcomp_para.add_on(i, j, - rate * dev);
		}
	}

	return;
}












//============================================ abstract functions =================================================
// func: multiplay the var array with the parameter matrix to get the result
// input: variable array; parameter matrix
// output: result variable array
void multi_array_matrix(float * input, Matrix matrix_para, float * result)
{
	long int dimension1 = matrix_para.get_dimension1();
	long int dimension2 = matrix_para.get_dimension2();

	for(long int i=0; i< dimension1; i++)
	{
		result[i] = 0;
		for(long int j=0; j<dimension2-1; j++)
		{
			float para = matrix_para.get(i, j);
			result[i] += input[j] * para;
		}
		result[i] += 1 * matrix_para.get(i, dimension2 - 1);		// here we do have the regression intercept term
	}

	return;
}



// func: multiply the SNP array with the cis- SNP parameter matrix, to get the expression list
// TODO: this might need to be changed, as I don't want to bring the global variables in this routine
void multi_array_matrix_imcomp(array<float *, NUM_CHR> * input_pointer, Matrix_imcomp matrix_imcomp_para, float * result)
{

	/*
	// DEBUG: check the genotyoe list
	for(int j=0; j<NUM_CHR; j++)  // across all the chromosomes
	{
		for(long k=0; k<snp_name_list[j].size(); k++)			// TODO: this is to be corrected, as we don't want to see global variables here
		{
			float var = (*input_pointer)[j][k];
			cout << var << "\t";
		}
	}
	cout << endl;
	// DEBUG done: there is no problem
	*/


	long int dimension1 = matrix_imcomp_para.get_dimension1();
	for(long int i=0; i<dimension1; i++)
	{
		int chr = matrix_imcomp_para.get_chr(i);
		long int start = matrix_imcomp_para.get_sst(i);
		long int dimension2 = matrix_imcomp_para.get_dimension2(i);

		result[i] = 0;
		for(long int j=0; j<dimension2; j++)
		{
			if(j == dimension2 - 1)
			{
				float par = matrix_imcomp_para.get(i, j);
				result[i] += 1 * par;			// the last one in the parameter list is for the intercept term
			}
			else
			{
				long int pos = start + j;
				float var = (*input_pointer)[chr-1][pos];
				float par = matrix_imcomp_para.get(i, j);
				result[i] += var * par;
			}
		}



		/*
		// DEBUG: let's also save the two lists (var and par), for the problem items
		// DEBUG
		if(i == 490 or i == 1057 or i == 489)
		{
			cout << result[i] << "\t" << chr << "\t" << dimension2 << endl;

			// DEBUG
			cout << snp_name_list[0].size() << endl;


			// DEBUG
			for(long k=0; k<snp_name_list[0].size(); k++)			// TODO: this is to be corrected, as we don't want to see global variables here
			{
				float var = (*input_pointer)[chr-1][k];
				cout << var << "\t";
			}
			cout << "\n";


			// enumerate all the elements (var, par)
			for(long int j = 0; j < dimension2; j++)
			{
				if(j == dimension2 - 1)
				{
					float par = matrix_imcomp_para.get(i, j);
					cout << "1 " << par << "\t";
				}
				else
				{
					long int pos = start + j;
					float var = (*input_pointer)[chr-1][pos];
					float par = matrix_imcomp_para.get(i, j);
					cout << var << " " << par << " " << j << " " << pos << "\t";
				}
			}
			cout << "\n";
		}
		// DEBUG: the genotype calling has some problem (some genotype are quite large; but others are all 0)
	*/



	}



	return;
}



// func: multiply the SNP array list with the parameter matrix
void multi_array_list_matrix(array<float *, NUM_CHR> * input_pointer, Matrix matrix_para, float * result)
{
	long int dimension1 = matrix_para.get_dimension1();
	long int dimension2 = matrix_para.get_dimension2();
	for(int i=0; i<dimension1; i++)
	{
		result[i] = 0;
		long int count = 0;
		for(int j=0; j<NUM_CHR; j++)  // across all the chromosomes
		{
			for(long k=0; k<snp_name_list[j].size(); k++)			// TODO: this is to be corrected, as we don't want to see global variables here
			{
				float var = (*input_pointer)[j][k];
				float par = matrix_para.get(i, count);
				result[i] += var * par;
				count ++;
			}
		}
		// we have not yet considered the intercept term by now
		float par = matrix_para.get(i, dimension2 - 1);				// we do have the intercept term here
		result[i] += par;
	}

	return;
}





//============================================ back propogation =================================================
// NOTE: this is an additive function that update the studied parameter derivatives
// we have several types of backpropogation:
// direct association, imcomplete parameter matrix 				[cis- association]
// last layer, complete parameter matrix 						[hidden batch to genes, cellenv to genes]
// inter layer, complete parameter matrix 						[snp to cellenv, batch to hidden batch]


// this is currently specially for cis- association:
// we usually have only direct regulation from cis- regulators to the genes (this is the current setting)
// pseudo: (expected rpkm - real rpkm) * genotype
void backward_error_prop_direct_imcomp(Matrix_imcomp matrix_imcomp_para_dev, float * error_list, array<float *, NUM_CHR> * input_list_pointer)
{


	// DEBUG
	/*
	cout << "$$" << endl;
	for(long int i=0; i<snp_name_list[0].size(); i++)
	{
		float dosage = (*input_list_pointer)[0][i];
		if(isnan(dosage))
		{
			cout << i << " " << dosage << endl;
		}
	}
	*/

	/*
	// DEBUG: save the genotype of this individual
	char filename[100] = "../result_tempdata/genotype.txt";
	// char filename[100] = "../result/para_cis_gene/";
	// char temp[10];
	// sprintf(temp, "%d", i+1);
	// strcat(filename, "etissue");
	// strcat(filename, temp);
	// strcat(filename, ".txt");

	FILE * file_out = fopen(filename, "w+");
	if(file_out == NULL)
	{
	    fputs("File error\n", stderr); exit(1);
	}

	int count = 0;
	for(int i=0; i<etissue_list.size(); i++)
	{
		count ++;
		string etissue = etissue_list[i];
		char buf[100];
		sprintf(buf, "%s\t%d\n", etissue.c_str(), count);
		fwrite(buf, sizeof(char), strlen(buf), file_out);
	}
	fclose(file_out);
	*/





	long int dimension1 = matrix_imcomp_para_dev.get_dimension1();
	for(long int i=0; i<dimension1; i++)
	{
		//float diff = expr_con_pointer[i] - (*expr_list_pointer)[i];
		float error = error_list[i];

		int chr = matrix_imcomp_para_dev.get_chr(i);
		long int pos_start = matrix_imcomp_para_dev.get_sst(i);

		long int dimension2 = matrix_imcomp_para_dev.get_dimension2(i);
		for(long int j=0; j<dimension2; j++)
		{
			if(j == dimension2 - 1)								// we do have the intercept term here
			{
				matrix_imcomp_para_dev.add_on(i, j, 1 * error);




				/*
				// DEBUG: check whether or not a large number contributes to nan
				float value = matrix_imcomp_para_dev.get(i, j);
				if(isnan(value))
				{
					//cout << error << endl;
					cout << i << " " << j << endl;
				}
				*/




				break;
			}

			long int pos = pos_start + j;
			float value = (*input_list_pointer)[chr-1][pos];			// here is a bug!!! the list is indexed by "chr-1"!!!
			matrix_imcomp_para_dev.add_on(i, j, value * error);





			/*
			// DEBUG: check whether or not a large number contributes to nan
			value = matrix_imcomp_para_dev.get(i, j);
			if(isnan(value))
			{
				//cout << error << endl;
				cout << i << " " << j << " " << (*input_list_pointer)[chr-1][pos] << endl;
			}
			*/





		}
	}

	return;
}



// pseudo: (expected rpkm - real rpkm) * cell_env
// pseudo: (expected rpkm - real rpkm) * hidden batch
void backward_error_prop_last_layer(Matrix matrix_para_dev, float * error_list, float * input)
{
	long int dimension1 = matrix_para_dev.get_dimension1();
	long int dimension2 = matrix_para_dev.get_dimension2();

	for(long int i=0; i<dimension1; i++)
	{
		float error = error_list[i];
		for(long int j=0; j<dimension2; j++)
		{
			if(j == dimension2 - 1)								// we do have the intercept term
			{
				matrix_para_dev.add_on(i, j, 1 * error);
				break;
			}

			float value = input[j];
			matrix_para_dev.add_on(i, j, value * error);
		}
	}

	return;
}



// inter layer 1: first input layer is the linked list -- cis- parameters
// pseudo: [ \sum w3 * (expected rpkm - real rpkm) ] * g'(w2 * x1) * x1
// NOTE: the only difference between 1 and 2 is the input (list of lists, or simply list)
void backward_error_prop_inter_layer_1(float * error_list, Matrix matrix_para_second, Matrix matrix_para_dev_first, float * hidden, array<float *, NUM_CHR> * input_pointer)
{
	// // DEBUG: maybe I want to check the backpropogated errors, so I keep a sample here
	// //===========================================================================================
	// //===========================================================================================
 //    FILE * file_out1 = fopen("../temp_data/error_cellenv_1.txt", "w+");
 //    if(file_out1 == NULL)
 //    {
 //        fputs("File error\n", stderr); exit(1);
 //    }
	// FILE * file_out2 = fopen("../temp_data/error_cellenv_2.txt", "w+");
	// if(file_out2 == NULL)
	// {
	// 	fputs("File error\n", stderr); exit(1);
	// }
	// //===========================================================================================
	// //===========================================================================================

	// // DEBUG
	// char buf[1024];

	long int dimension2 = matrix_para_second.get_dimension2();
	for(long int i=0; i<dimension2 - 1; i++)							// the intercept term is not evaluated in back propogation
	{
		//
		long int dimension1 = matrix_para_second.get_dimension1();
		float temp = 0;
		for(int t=0; t<dimension1; t++)
		{
			float par = matrix_para_second.get(t, i);
			temp += par * error_list[t];
		}

		// // DEBUG
		// // save the back-propogated errors to the file, and check
		// sprintf(buf, "%f\n", temp);
		// fwrite(buf, sizeof(char), strlen(buf), file_out1);

		//
		temp *= neuralnet_ac_func_dev(hidden[i]);
		//

		// // DEBUG
		// // save the back-propogated errors to the file, and check
		// sprintf(buf, "%f\n", temp);
		// fwrite(buf, sizeof(char), strlen(buf), file_out2);

		long int dimension3 = matrix_para_dev_first.get_dimension2();
		long int count = 0;
		for(int j=0; j<NUM_CHR; j++)
		{
			for(long k=0; k<snp_name_list[j].size(); k++)				// TODO: we don't want global variables visiable here!!!
			{
				float var = (*input_pointer)[j][k];
				matrix_para_dev_first.add_on(i, count, var * temp);
				count ++;
			}
		}
		matrix_para_dev_first.add_on(i, dimension3 - 1, 1 * temp);		// we do have the intercept term here

	}

	// // DEBUG
	// //===========================================================================================
	// //===========================================================================================
	// fclose(file_out1);
	// fclose(file_out2);
	// //===========================================================================================
	// //===========================================================================================

	return;
}




// inter layer 2: first input layer is simply a list
// pseudo: [ \sum w5 * (expected rpkm - real rpkm) ] * g'(w4 * x2) * x2
void backward_error_prop_inter_layer_2(float * error_list, Matrix matrix_para_second, Matrix matrix_para_dev_first, float * hidden, float * input)
{
	long int dimension2 = matrix_para_second.get_dimension2();
	for(long int i=0; i<dimension2 - 1; i++)							// the intercept term is not evaluated in back propogation
	{
		long int dimension1 = matrix_para_second.get_dimension1();
		float temp = 0;
		for(int t=0; t<dimension1; t++)
		{
			float par = matrix_para_second.get(t, i);
			temp += par * error_list[t];
		}
		//
		temp *= neuralnet_ac_func_dev(hidden[i]);
		//
		long int dimension3 = matrix_para_dev_first.get_dimension2();
		for(long int j=0; j<dimension3; j++)
		{
			if(j == dimension3 - 1)										// we do have the intercept term here
			{
				matrix_para_dev_first.add_on(i, j, 1 * temp);
				break;
			}

			float var = input[j];
			matrix_para_dev_first.add_on(i, j, var * temp);
		}
	}

	return;
}


