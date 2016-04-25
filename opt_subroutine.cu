// subroutines of optimization procedure

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unordered_map>
#include <string>
#include <vector>
#include "basic.h"
#include <forward_list>
#include <utility>
#include "genotype.h"
#include "expression.h"
#include "global.h"
#include "main.h"  // typedef struct tuple_long
#include <math.h>       /* exp */
#include "opt_subroutine.h"
#include "optimization.h"
#include "opt_nn_acfunc.h"
#include "opt_debugger.h"
#include "libfunc_matrix.h"
#include <cmath>        // std::abs
// includes CUDA runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include "opt_subroutine_gpu.cuh"

#include <sys/time.h>
#include <time.h>       /* clock_t, clock, CLOCKS_PER_SEC */





using namespace std;






// forward and backward propagation for one mini-batch
void forward_backward_prop_batch(string etissue, int pos_start, int num_esample)
{
	cout << "[@@] entering the forward-backward propagation..." << endl;

	int etissue_index = etissue_index_map[etissue];
	long int dimension1;
	long int dimension2;



	/*
	// DEBUG:
	// add one test: test whether the GPU actually touches the transfered data, thus add one to the data
	gpu_addone<<<( num_para_cis+255 )/256 , 256 >>>( num_para_cis , d_list_para_cis_gene[etissue_index]);

	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
	gpu_addone<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_snp_cellenv);

	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
	gpu_addone<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_list_para_cellenv_gene[etissue_index]);

	dimension1 = matrix_para_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_batch_batch_hidden.get_dimension2();
	gpu_addone<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_batch_batch_hidden);

	dimension1 = matrix_para_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_batch_hidden_gene.get_dimension2();
	gpu_addone<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_batch_hidden_gene);

	return;
	*/




	//******************* initialize all the parameter derivatives (as 0) *******************
	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	// vector<Matrix_imcomp> cube_para_dev_cis_gene;
	cube_para_dev_cis_gene[etissue_index].clean();

	// Matrix matrix_para_dev_snp_cellenv;
	matrix_para_dev_snp_cellenv.clean();

	// vector<Matrix> matrix_para_dev_cellenv_gene;
	cube_para_dev_cellenv_gene[etissue_index].clean();

	// Matrix matrix_para_dev_batch_batch_hidden;
	matrix_para_dev_batch_batch_hidden.clean();

	// Matrix matrix_para_dev_batch_hidden_gene;
	matrix_para_dev_batch_hidden_gene.clean();
	*/



	//============== timing starts ================
    struct timeval time_start;
    struct timeval time_end;
    double diff;
    gettimeofday(&time_start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	// clean GPU memory for para_dev (as the forward_backward function is additive)
	gpu_clean<<<( num_para_cis+255 )/256 , 256 >>>( num_para_cis , d_list_para_dev_cis_gene[etissue_index]);

	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	gpu_clean<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_dev_snp_cellenv);

	//vector<Matrix> cube_para_dev_cellenv_gene --> vector<float *> d_list_para_dev_cellenv_gene
	dimension1 = cube_para_dev_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[etissue_index].get_dimension2();
	float * d_para_dev_cellenv_gene = d_list_para_dev_cellenv_gene[etissue_index];
	gpu_clean<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_dev_cellenv_gene);

	//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
	gpu_clean<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_dev_batch_batch_hidden);

	//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
	gpu_clean<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_dev_batch_hidden_gene);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);





	//****************************** enter the mini-batch ***********************************
	cout << "we are entering a new mini-batch..." << endl;
	for(int count=0; count<batch_size; count++)
	{



		// TODO: I will save the time on dosage loading/transmitting, and expression loading and transmitting
		// TODO:
		// 1. float * d_batch_var
		// 2. float * d_expr


		//==========================================================================================================
		//========================================== CPU data preparation ==========================================
		//==========================================================================================================
		//============== timing starts ================
    	struct timeval time_start;
    	struct timeval time_end;
    	double diff;
    	gettimeofday(&time_start, NULL);


		int pos = (pos_start + count) % (num_esample);
		string esample = esample_tissue_rep[etissue][pos];
		string individual = sample_to_individual(esample);
		cout << "======== current sample #" << pos+1 << ": " << esample << endl;

		//=================================================== init ============================================================
		// get the: 0. esample and individual; 1. genotype; 2. expression data; 3. batch variables
		// to: 1. forward_backward propagation;
		// genotype dosage data
		//cout << "getting the dosage data for individual #" << individual << endl;

		/*
		snp_dosage_load(&snp_dosage_list, individual);  // snp dosage data for one individual across all chromosomes
		*/

		// expression rpkm data: eQTL_tissue_rep[etissue][esample]
		//cout << "we have this amount of genes expressed in this individual:" << eQTL_tissue_rep[etissue][esample].size() << endl;
		// and the batch variable for this individual and this sample

		//============== timing ends ================
		gettimeofday(&time_end, NULL);
		diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
		printf("pre- Time used totally is %f seconds.\n", diff);


		/*
		//============== timing starts ================
		gettimeofday(&time_start, NULL);

		int num_batch_individual = batch_individual[individual].size();
		int index = 0;
		for(int i=0; i<num_batch_individual; i++)
		{
			float value = batch_individual[individual][i];
			batch_var[index] = value;
			index++;
		}
		int num_batch_sample = batch_sample[esample].size();
		for(int i=0; i<num_batch_sample; i++)
		{
			float value = batch_sample[esample][i];
			batch_var[index] = value;
			index++;
		}

		//============== timing ends ================
		gettimeofday(&time_end, NULL);
		diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
		printf("batch CPU: Time used totally is %f seconds.\n", diff);
		*/




		//==========================================================================================================
		//========================================== CPU data preparation ==========================================
		//==========================================================================================================
		//============== timing starts ================
		gettimeofday(&time_start, NULL);

		// TODO GPU: load everything into GPU memory, and make them addressable
		// NOTE: CPU should have all the data, and GPU has also another copy of these data
		//===============================================
		//================ GPU computing ================
		//===============================================
		// //==== int * d_etissue_index_p
		// checkCudaErrors(cudaMemcpy( d_etissue_index_p, &etissue_index, 1*sizeof(int), cudaMemcpyHostToDevice));

    	/*
		//==== float * d_snp
		long int pos_start = 0;
		for(int i=0; i<NUM_CHR; i++)  // across all the chromosomes
		{
			long int dimension = snp_name_list[i].size();
			checkCudaErrors(cudaMemcpy( (d_snp + pos_start), snp_dosage_list[i], dimension*sizeof(float), cudaMemcpyHostToDevice));
			pos_start += dimension;
		}
		*/

		int snp_index = d_snp_index_map[individual];
		d_snp = d_snp_list[snp_index];

		//============== timing ends ================
		gettimeofday(&time_end, NULL);
		diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
		printf("dosage GPU: Time used totally is %f seconds.\n", diff);




		//============== timing starts ================
		gettimeofday(&time_start, NULL);

		/*
		//==== float * d_batch_var
		checkCudaErrors(cudaMemcpy( d_batch_var, batch_var, num_batch*sizeof(float), cudaMemcpyHostToDevice));
		*/

		int batch_index = d_batch_index_map[esample];
		d_batch_var = d_batch_list[batch_index];

		//============== timing ends ================
		gettimeofday(&time_end, NULL);
		diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
		printf("batch GPU: Time used totally is %f seconds.\n", diff);





		//============== timing starts ================
    	gettimeofday(&time_start, NULL);

    	/*
		//==== float * d_expr
		float * expr_list = (float *)malloc(num_gene*sizeof(float));
		for(int i=0; i<num_gene; i++)
		{
			expr_list[i] = eQTL_tissue_rep[etissue][esample][i];
		}
		checkCudaErrors(cudaMemcpy( d_expr, expr_list, num_gene*sizeof(float), cudaMemcpyHostToDevice));
		free(expr_list);
		*/

		int sample_index = d_sample_index_map[esample];
		d_expr = d_sample_list[sample_index];

		//============== timing ends ================
		gettimeofday(&time_end, NULL);
		diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
		printf("expr: Time used totally is %f seconds.\n", diff);







		//============== timing starts ================
    	gettimeofday(&time_start, NULL);


		forward_backward(etissue_index,
						&snp_dosage_list,
						&eQTL_tissue_rep[etissue][esample],

						gene_rpkm_exp,
						cellenv_hidden_var,
						batch_var,
						batch_hidden_var,

						cube_para_dev_cis_gene[etissue_index],
						matrix_para_dev_snp_cellenv,
						cube_para_dev_cellenv_gene[etissue_index],
						matrix_para_dev_batch_batch_hidden,
						matrix_para_dev_batch_hidden_gene
						);


		//============== timing ends ================
		gettimeofday(&time_end, NULL);
		diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
		printf("(forward_backward) Time used totally is %f seconds.\n", diff);





		// DEBUG
		// for this sample (or for this individual as we fix the tissue type currenlty):
		// I want to check the genotype of this individual, as the problem seems to come up there!!!
		// save: individual ID; genotype
		// debug here and also in the subroutine
		// for(long int i=0; i<snp_name_list[0].size(); i++)
		// {
		// 	float dosage = snp_dosage_list[0][i];
		// 	if(isnan(dosage))
		// 	{
		// 		cout << individual << " " << i << " " << dosage << endl;
		// 	}
		// }


		// iterating in this mini-batch
	}



	/*
	// DEBUG: debug the parameter dev, and the current cellenv and hiddenbatch
	para_temp_save_dev(etissue_index);

	// DEBUG: save some variables (is this so necessary especially after a whole batch?)
	char filename[100] = "../result_tempdata/var_cellenv.txt";
	para_temp_save_var(cellenv_hidden_var, num_cellenv, filename);
	sprintf(filename, "%s", "../result_tempdata/var_batch_hidden.txt");
	para_temp_save_var(batch_hidden_var, num_batch_hidden, filename);
	*/




	// DEBUG
    // DEBUG: test the dev here
    // transfer them back, and save into temp files
    // if this is working, the problem is below; else, I can start debugging the above
	//==== para_dev transfer back
	//
    for(int i=0; i<num_etissue; i++)
    {
		float * d_para_dev_cis_gene = d_list_para_dev_cis_gene[i];
		long int dimension1 = cube_para_dev_cis_gene[i].get_dimension1();
		long int pos_start = 0;
		for(long int j=0; j<dimension1; j++)
		{
			long int amount = cube_para_dev_cis_gene[i].get_dimension2(j);
			float * x = cube_para_dev_cis_gene[i].get_list(j);
			checkCudaErrors(cudaMemcpy(x, (d_para_dev_cis_gene + pos_start), amount*sizeof(float), cudaMemcpyDeviceToHost));
			pos_start += amount;
		}
    }

    //
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_snp_cellenv + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

    //
	dimension1 = cube_para_dev_cellenv_gene[0].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[0].get_dimension2();
	for(int j=0; j<num_etissue; j++)
	{
		float * d_para_dev_cellenv_gene = d_list_para_dev_cellenv_gene[j];
	    for(long int i=0; i<dimension1; i++)
	    {
	    	float * x = cube_para_dev_cellenv_gene[j].get_list(i);
	    	long int pos_start = i * dimension2;
			checkCudaErrors(cudaMemcpy(x, (d_para_dev_cellenv_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
	    }
	}

	//
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_batch_hidden.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_batch_batch_hidden + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

    //
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_hidden_gene.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_batch_hidden_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }


	//==== para_dev save
	char filename[100];
	//================================ vector<Matrix_imcomp> cube_para_dev_cis_gene ================================
	// this is tissue specific
	sprintf(filename, "%s", "../result_tempdata/para_dev_cis_gene_before.txt");
	cube_para_dev_cis_gene[etissue_index].save(filename);


	//================================== Matrix matrix_para_dev_snp_cellenv ===================================
	sprintf(filename, "%s", "../result_tempdata/para_dev_snp_cellenv_before.txt");
	matrix_para_dev_snp_cellenv.save(filename);


	//============================== vector<Matrix> cube_para_dev_cellenv_gene ==============================
	// this is tissue specific
	sprintf(filename, "%s", "../result_tempdata/para_dev_cellenv_gene_before.txt");
	cube_para_dev_cellenv_gene[etissue_index].save(filename);


	//=============================== Matrix matrix_para_dev_batch_batch_hidden ===============================
	sprintf(filename, "%s", "../result_tempdata/para_dev_batch_batch_hidden_before.txt");
	matrix_para_dev_batch_batch_hidden.save(filename);


	//=============================== Matrix matrix_para_dev_batch_hidden_gene ================================
	sprintf(filename, "%s", "../result_tempdata/para_dev_batch_hidden_gene_before.txt");
	matrix_para_dev_batch_hidden_gene.save(filename);







	// SHOULD START DEBUGGING FROM HERE !!!
	// the previous routines seem problematic, and the following routines also bad









	//********************************* aggregation of this mini-batch *****************************************
	// 1. average the derivatives calculated from previous steps
	// 2. will add the derivatives due to regularization in the next part
	cout << "aggregation of this mini-batch..." << endl;

	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	// vector<Matrix_imcomp> cube_para_dev_cis_gene;		// NOTE: won't transform this into GPU code
	cube_para_dev_cis_gene[etissue_index].scale( 1.0 / batch_size );
	//matrix_para_dev_snp_cellenv.scale( 1.0 / batch_size );
	matrix_para_dev_snp_cellenv.scale( 1.0 / batch_size );
	// vector<Matrix> cube_para_dev_cellenv_gene;
	cube_para_dev_cellenv_gene[etissue_index].scale( 1.0 / batch_size );
	// Matrix matrix_para_dev_batch_batch_hidden;
	matrix_para_dev_batch_batch_hidden.scale( 1.0 / batch_size );
	// Matrix matrix_para_dev_batch_hidden_gene;
	matrix_para_dev_batch_hidden_gene.scale( 1.0 / batch_size );
	*/

	//============== timing starts ================
    gettimeofday(&time_start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	float factor = 1.0 / batch_size;

	gpu_scale<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis , factor, d_list_para_dev_cis_gene[etissue_index]);

	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	gpu_scale<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , factor, d_para_dev_snp_cellenv);

	//vector<Matrix> cube_para_dev_cellenv_gene --> vector<float *> d_list_para_dev_cellenv_gene
	dimension1 = cube_para_dev_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[etissue_index].get_dimension2();
	gpu_scale<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , factor, d_list_para_dev_cellenv_gene[etissue_index]);

	//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
	gpu_scale<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , factor, d_para_dev_batch_batch_hidden);

	//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
	gpu_scale<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , factor, d_para_dev_batch_hidden_gene);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);



	/*
	// DEBUG
	//
    for(int i=0; i<num_etissue; i++)
    {
		float * d_para_dev_cis_gene = d_list_para_dev_cis_gene[i];
		long int dimension1 = cube_para_dev_cis_gene[i].get_dimension1();
		long int pos_start = 0;
		for(long int j=0; j<dimension1; j++)
		{
			long int amount = cube_para_dev_cis_gene[i].get_dimension2(j);
			float * x = cube_para_dev_cis_gene[i].get_list(j);
			checkCudaErrors(cudaMemcpy(x, (d_para_dev_cis_gene + pos_start), amount*sizeof(float), cudaMemcpyDeviceToHost));
			pos_start += amount;
		}
    }

    //
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_snp_cellenv + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

    //
	dimension1 = cube_para_dev_cellenv_gene[0].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[0].get_dimension2();
	for(int j=0; j<num_etissue; j++)
	{
		float * d_para_dev_cellenv_gene = d_list_para_dev_cellenv_gene[j];
	    for(long int i=0; i<dimension1; i++)
	    {
	    	float * x = cube_para_dev_cellenv_gene[j].get_list(i);
	    	long int pos_start = i * dimension2;
			checkCudaErrors(cudaMemcpy(x, (d_para_dev_cellenv_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
	    }
	}

	//
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_batch_hidden.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_batch_batch_hidden + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

    //
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_hidden_gene.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_batch_hidden_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

	//================================ vector<Matrix_imcomp> cube_para_dev_cis_gene ================================
	// this is tissue specific
	sprintf(filename, "%s", "../result_tempdata/para_dev_cis_gene_after.txt");
	cube_para_dev_cis_gene[etissue_index].save(filename);


	//================================== Matrix matrix_para_dev_snp_cellenv ===================================
	sprintf(filename, "%s", "../result_tempdata/para_dev_snp_cellenv_after.txt");
	matrix_para_dev_snp_cellenv.save(filename);


	//============================== vector<Matrix> cube_para_dev_cellenv_gene ==============================
	// this is tissue specific
	sprintf(filename, "%s", "../result_tempdata/para_dev_cellenv_gene_after.txt");
	cube_para_dev_cellenv_gene[etissue_index].save(filename);


	//=============================== Matrix matrix_para_dev_batch_batch_hidden ===============================
	sprintf(filename, "%s", "../result_tempdata/para_dev_batch_batch_hidden_after.txt");
	matrix_para_dev_batch_batch_hidden.save(filename);


	//=============================== Matrix matrix_para_dev_batch_hidden_gene ================================
	sprintf(filename, "%s", "../result_tempdata/para_dev_batch_hidden_gene_after.txt");
	matrix_para_dev_batch_hidden_gene.save(filename);
	*/










    /*
    //==========================================================
    //======== GPU testing routine: TODO-OR-NOT testing ========
    //==========================================================
	//============== timing starts ================
    struct timeval time_start;
    struct timeval time_end;
    double diff;
    gettimeofday(&time_start, NULL);

	matrix_para_dev_snp_cellenv.scale( 1.0 / batch_size );

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);

    printf("========================================================================================\n");
    printf("start testing the GPU devices...\n");
	long int dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	//matrix_para_dev_snp_cellenv.scale( 1.0 / batch_size );

	float factor = 1.0 / batch_size;

    int deviceID = 0;
    checkCudaErrors(cudaSetDevice(deviceID));

    float * d_x;
    checkCudaErrors(cudaMalloc(&d_x, (dimension1*dimension2)*sizeof(float)));

    // memory copy
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
    	checkCudaErrors(cudaMemcpy( (d_x + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }

	//============== timing starts ================
    gettimeofday(&time_start, NULL);

	gpu_scale<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , factor, d_x);
	//checkCudaErrors(cudaMemcpy(x, d_x, dimension2*sizeof(float), cudaMemcpyDeviceToHost));

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);

    checkCudaErrors(cudaFree(d_x));

    checkCudaErrors(cudaDeviceReset());
    printf("========================================================================================\n");
    */





	/*
    //====================================================
    //======== GPU testing routine: speed testing ========
    //====================================================
	// (Apr.17) working on "matrix_para_dev_snp_cellenv"
	// Matrix matrix_para_dev_snp_cellenv;
	//============== timing starts ================
    struct timeval time_start;
    struct timeval time_end;
    double diff;
    gettimeofday(&time_start, NULL);

	matrix_para_dev_snp_cellenv.scale( 1.0 / batch_size );

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);


    printf("========================================================================================\n");
    printf("start testing the GPU devices...\n");
	long int dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	//matrix_para_dev_snp_cellenv.scale( 1.0 / batch_size );

	float factor = 1.0 / batch_size;

    int deviceID = 0;
    checkCudaErrors(cudaSetDevice(deviceID));

    float * d_x;
    checkCudaErrors(cudaMalloc(&d_x, dimension2*sizeof(float)));


	//============== timing starts ================
    gettimeofday(&time_start, NULL);


    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_snp_cellenv.get_list(i);
    	checkCudaErrors(cudaMemcpy(d_x, x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
	    gpu_scale<<<(dimension2+255)/256, 256>>>(dimension2, factor, d_x);
	    checkCudaErrors(cudaMemcpy(x, d_x, dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);

    checkCudaErrors(cudaFree(d_x));

    checkCudaErrors(cudaDeviceReset());
    printf("========================================================================================\n");
    */



    //========================================================
    //======== independent GPU testing routine: saxpy ========
    //========================================================
    /*
    printf("========================================================================================\n");
    printf("start testing the GPU devices...\n");

    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));
    printf("DevicecheckCudaErrors Count: %d\n", deviceCount);

    //int deviceID = findCudaDevice(argc, argv);
    int deviceID = 0;
    cudaDeviceProp prop;
    checkCudaErrors(cudaGetDeviceProperties(&prop, deviceID));
    if (prop.major < 2)
    {
        printf("Quit: current GPU device has compute SM%d.%d, Exiting...", prop.major, prop.minor);
        exit(EXIT_WAIVED);
    }

    printf("Device %d: \"%s\" with Compute %d.%d capability\n",
           deviceID, prop.name, prop.major, prop.minor);
    checkCudaErrors(cudaSetDevice(deviceID));

    //==== memory allocating
    int N1 = 1<<20;
    cout << N1 << endl;
    float *x, *y, *d_x, *d_y;

    x = (float*)malloc(N1*sizeof(float));
    y = (float*)malloc(N1*sizeof(float));

    checkCudaErrors(cudaMalloc(&d_x, N1*sizeof(float)));
    checkCudaErrors(cudaMalloc(&d_y, N1*sizeof(float)));

    for (int i = 0; i < N1; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    checkCudaErrors(cudaMemcpy(d_x, x, N1*sizeof(float), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_y, y, N1*sizeof(float), cudaMemcpyHostToDevice));


    //==== Perform SAXPY on 1M elements
    //saxpy<<<(N1+255)/256, 256>>>(N1, 2.0f, d_x, d_y);     // the classical one
    saxpy<<<(N1+255)/256, 256>>>(N1, 2.0f, d_x, d_y);

    checkCudaErrors(cudaMemcpy(y, d_y, N1*sizeof(float), cudaMemcpyDeviceToHost));

    float maxError = 0.0f;
    for(int i = 0; i < N1; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
    printf("Max error (saxpy): %f\n", maxError);

    free(x);
    free(y);
    checkCudaErrors(cudaFree(d_x));
    checkCudaErrors(cudaFree(d_y));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    checkCudaErrors(cudaDeviceReset());
    printf("========================================================================================\n");
	*/







    // DEBUG
    /*
    // DEBUG: test the dev here
    // transfer them back, and save into temp files
    // if this is working, the problem is below; else, I can start debugging the above
	//==== para_dev transfer back
	//
    for(int i=0; i<num_etissue; i++)
    {
		float * d_para_dev_cis_gene = d_list_para_dev_cis_gene[i];
		long int dimension1 = cube_para_dev_cis_gene[i].get_dimension1();
		long int pos_start = 0;
		for(long int j=0; j<dimension1; j++)
		{
			long int amount = cube_para_dev_cis_gene[i].get_dimension2(j);
			float * x = cube_para_dev_cis_gene[i].get_list(j);
			checkCudaErrors(cudaMemcpy(x, (d_para_dev_cis_gene + pos_start), amount*sizeof(float), cudaMemcpyDeviceToHost));
			pos_start += amount;
		}
    }

    //
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_snp_cellenv + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

    //
	dimension1 = cube_para_dev_cellenv_gene[0].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[0].get_dimension2();
	for(int j=0; j<num_etissue; j++)
	{
		float * d_para_dev_cellenv_gene = d_list_para_dev_cellenv_gene[j];
	    for(long int i=0; i<dimension1; i++)
	    {
	    	float * x = cube_para_dev_cellenv_gene[j].get_list(i);
	    	long int pos_start = i * dimension2;
			checkCudaErrors(cudaMemcpy(x, (d_para_dev_cellenv_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
	    }
	}

	//
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_batch_hidden.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_batch_batch_hidden + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

    //
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_batch_hidden_gene.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_batch_hidden_gene + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }



	//==== para_dev save
	//================================ vector<Matrix_imcomp> cube_para_cis_gene ================================
	// this is tissue specific
	char filename[100] = "../result_tempdata/para_dev_cis_gene.txt";
	cube_para_dev_cis_gene[etissue_index].save(filename);


	//================================== Matrix matrix_para_snp_cellenv ===================================
	sprintf(filename, "%s", "../result_tempdata/para_dev_snp_cellenv.txt");
	matrix_para_dev_snp_cellenv.save(filename);


	//============================== vector<Matrix> cube_para_cellenv_gene ==============================
	// this is tissue specific
	sprintf(filename, "%s", "../result_tempdata/para_dev_cellenv_gene.txt");
	cube_para_dev_cellenv_gene[etissue_index].save(filename);


	//=============================== Matrix matrix_para_batch_batch_hidden ===============================
	sprintf(filename, "%s", "../result_tempdata/para_dev_batch_batch_hidden.txt");
	matrix_para_dev_batch_batch_hidden.save(filename);


	//=============================== Matrix matrix_para_batch_hidden_gene ================================
	sprintf(filename, "%s", "../result_tempdata/para_dev_batch_hidden_gene.txt");
	matrix_para_dev_batch_hidden_gene.save(filename);



	// So ,it start going wrong from above
	*/












	//============== timing starts ================
    gettimeofday(&time_start, NULL);

	//===================================== Regularization in Regression =====================================
	// NOTE: not yet tested, but this is too straightforward to test
	regularization(etissue_index);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("regularization: Time used totally is %f seconds.\n", diff);





	//============== timing starts ================
    gettimeofday(&time_start, NULL);

	//=========================================== Gradient Descent ===========================================
	// NOTE: this routine is tested to be correct (Apr.25)
	gradient_descent(etissue_index);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("gd: Time used totally is %f seconds.\n", diff);









	cout << "[@@] leaving the forward-backward propagation..." << endl;

}






// NOTE: ADDITIVE
// property of this function: additive to the total derivative after this round (additive)
// what we need for the following routine:
// dosage list; expression value list; expression list; cellenv list; batch list; batch hidden list; ALL parameter (derivative) containers
void forward_backward(int etissue_index,
	array<float *, NUM_CHR> * dosage_list_pointer,
	vector<float> * expr_list_pointer,

	float * expr_con_pointer,
	float * cellenv_con_pointer,
	float * batch_list_pointer,
	float * batch_hidden_con_pointer,
	// the new Matrix/Matrix_imcomp classes:
	Matrix_imcomp matrix_imcomp_para_dev_cis_gene,				// drop the Matrix_imcomp object, other than the full cube. TODO: not sure whether there are problems
	Matrix matrix_para_dev_snp_cellenv,
	Matrix matrix_para_dev_cellenv_gene,						// drop the Matrix object, other than the full cube
	Matrix matrix_para_dev_batch_batch_hidden,
	Matrix matrix_para_dev_batch_hidden_gene
	)
{


	// how to map these pointers:
	/*
	dosage_list_pointer --> &snp_dosage_list
	expr_list_pointer --> &eQTL_tissue_rep[etissue][esample]
	batch_list_pointer --> batch_var
	expr_con_pointer  --> gene_rpkm_exp
	cellenv_con_pointer --> cellenv_hidden_var
	batch_hidden_con_pointer --> batch_hidden_var

	// all the other Matrix/Matrix_imcomp are directly called
	*/


	// DEBUG: check the Nan in genotype
	/*
	cout << "$$$" << endl;
	for(long int i=0; i<snp_name_list[0].size(); i++)
	{
		float dosage = snp_dosage_list[0][i];
		if(isnan(dosage))
		{
			cout << i << " " << dosage << endl;
		}
	}
	*/





	// DEBUG: this is used to debug the program
	//char filename[100];




	/*
	// DEBUG: check the genotyoe list
	for(int j=0; j<NUM_CHR; j++)  // across all the chromosomes
	{
		for(long k=0; k<snp_name_list[j].size(); k++)			// TODO: this is to be corrected, as we don't want to see global variables here
		{
			float var = (*dosage_list_pointer)[j][k];
			cout << var << "\t";
		}
	}
	cout << endl;
	// DEBUG done: there is no problem
	*/




	//============== timing ================
	struct timeval time_start;
	struct timeval time_end;
	double diff;




	//========================================================================
	// two step: forward propagation (get the function values); backward propagation (get the parameter derivatives)
	//========================================================================
	//========================================================================
	// step#1: forward-propogation (cis-; cell env; batch)
	//========================================================================
	//========================================================================
	// ****************************** [part1] cis- *********************************
	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	// for cis-, two issues:
	// 1. if this is a XYMT gene, we don't have signal from it's cis- SNPs (not consider currently);
	// 2. we use (gene_cis_index[gene].second - gene_cis_index[gene].first + 1) as the length of the cis- parameter array
	float * expr_con_pointer_cis = (float *)calloc( num_gene, sizeof(float) );
	multi_array_matrix_imcomp(dosage_list_pointer, cube_para_cis_gene[etissue_index], expr_con_pointer_cis);

	// // DEBUG mode: let's save the signal from all the three pathways
	// sprintf(filename, "%s", "../result_tempdata/var_expr_cis.txt");
	// para_temp_save_var(expr_con_pointer_cis, num_gene, filename);
	*/

	//============== timing starts ================
	gettimeofday(&time_start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	//== multiply the input to the matrix
	gpu_matrix_mul_cis_mul<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis, num_gene, d_snp, d_list_para_cis_gene[etissue_index], d_temp_cis_gene,\
							d_cis_para_start, d_cis_para_amount, d_cis_snp_start, d_cis_para_index1);
	//== sum matrix
	gpu_matrix_mul_cis_add<<<(num_gene + 255) / 256 , 256 >>>( num_gene, d_temp_cis_gene, d_gene_rpkm_exp_cis,\
							d_cis_para_start, d_cis_para_amount, d_cis_snp_start);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("cis: Time used totally is %f seconds.\n", diff);






	// ********************* [part2] cell env relevant parameters *********************
	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	// from snp to cell env variables
	float * expr_con_pointer_cellenv = (float *)calloc( num_gene, sizeof(float) );
	multi_array_list_matrix(dosage_list_pointer, matrix_para_snp_cellenv, cellenv_con_pointer);

	// // // DEBUG
	// sprintf(filename, "%s", "../result_tempdata/var_cellenv_before.txt");
	// para_temp_save_var(cellenv_con_pointer, num_cellenv, filename);

	//$$$$$$$$$$$ perform the activation function here (logistic or something else) $$$$$$$$$$$$
	neuralnet_ac_func(cellenv_con_pointer, num_cellenv);

	// // // DEBUG
	// sprintf(filename, "%s", "../result_tempdata/var_cellenv_after.txt");
	// para_temp_save_var(cellenv_con_pointer, num_cellenv, filename);

	// from cell env variables to genes
	multi_array_matrix(cellenv_con_pointer, cube_para_cellenv_gene[etissue_index], expr_con_pointer_cellenv);

	// // // DEBUG
	// sprintf(filename, "%s", "../result_tempdata/var_expr_cellenv.txt");
	// para_temp_save_var(expr_con_pointer_cellenv, num_gene, filename);
	*/


	//============== timing starts ================
	gettimeofday(&time_start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	//==== from SNP to cellenv
	long int dimension1 = matrix_para_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_snp_cellenv.get_dimension2();
	//== multiply the input to the matrix
	gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_snp, d_para_snp_cellenv, d_temp_snp_cellenv);
	//== sum matrix
	gpu_matrix_mul_add<<<(dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_snp_cellenv, d_cellenv_hidden_var);

	//==== neuralnet
	gpu_neuralnet_ac_func<<<(num_cellenv + 255) / 256 , 256 >>>( num_cellenv , d_cellenv_hidden_var);

	//==== from cellenv to gene expression
	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
	//== multiply the input to the matrix
	float * d_para_cellenv_gene = d_list_para_cellenv_gene[etissue_index];
	gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_cellenv_hidden_var, d_para_cellenv_gene, d_temp_cellenv_gene);
	//== sum matrix
	gpu_matrix_mul_add<<< (dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_cellenv_gene, d_gene_rpkm_exp_cellenv);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("cellenv: Time used totally is %f seconds.\n", diff);





	// ********************* [part3] linear or non-linear batches *********************
	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	float * expr_con_pointer_batch = (float *)calloc( num_gene, sizeof(float) );
	// from original batch to hidden batch
	multi_array_matrix(batch_list_pointer, matrix_para_batch_batch_hidden, batch_hidden_con_pointer);

	// // DEBUG
	// sprintf(filename, "%s", "../result_tempdata/var_batch_hidden_before.txt");
	// para_temp_save_var(batch_hidden_con_pointer, num_batch_hidden, filename);

	//$$$$$$$$$$$ perform the activation function here (logistic or something else) $$$$$$$$$$$$
	neuralnet_ac_func(batch_hidden_con_pointer, num_batch_hidden);

	// // DEBUG
	// sprintf(filename, "%s", "../result_tempdata/var_batch_hidden_after.txt");
	// para_temp_save_var(batch_hidden_con_pointer, num_batch_hidden, filename);

	// from hidden batch to genes
	multi_array_matrix(batch_hidden_con_pointer, matrix_para_batch_hidden_gene, expr_con_pointer_batch);

	// // DEBUG
	// sprintf(filename, "%s", "../result_tempdata/var_expr_batch.txt");
	// para_temp_save_var(expr_con_pointer_batch, num_gene, filename);
	*/


	//============== timing starts ================
	gettimeofday(&time_start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	//==== from batch to batch_hidden
	dimension1 = matrix_para_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_batch_batch_hidden.get_dimension2();
	//== multiply the input to the matrix
	gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_batch_var, d_para_batch_batch_hidden, d_temp_batch_batch_hidden);
	//== sum matrix
	gpu_matrix_mul_add<<<(dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_batch_batch_hidden, d_batch_hidden_var);

	//==== neuralnet
	gpu_neuralnet_ac_func<<<(num_batch_hidden + 255) / 256 , 256 >>>( num_batch_hidden , d_batch_hidden_var);

	//==== from batch_hidden to gene
	dimension1 = matrix_para_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_batch_hidden_gene.get_dimension2();
	//== multiply the input to the matrix
	gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_batch_hidden_var, d_para_batch_hidden_gene, d_temp_batch_hidden_gene);
	//== sum matrix
	gpu_matrix_mul_add<<< (dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_batch_hidden_gene, d_gene_rpkm_exp_batch);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("batch: Time used totally is %f seconds.\n", diff);






	// ********************* [end] merge the signal from three pathways here, to expr_con_pointer *********************
	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	for(long int i=0; i<num_gene; i++)
	{
		expr_con_pointer[i] = expr_con_pointer_cis[i] + expr_con_pointer_cellenv[i] + expr_con_pointer_batch[i];
	}
	//free(expr_con_pointer_cis);
	//free(expr_con_pointer_cellenv);
	//free(expr_con_pointer_batch);

	// error is the thing actually needed
	//float * error_list = (float *)calloc(num_gene, sizeof(float));
	float * error_list = (float *)malloc(num_gene*sizeof(float));
	for(long int i=0; i<num_gene; i++)
	{
		error_list[i] = expr_con_pointer[i] - (*expr_list_pointer)[i];
	}
	*/

	/*
	//=======================================================================
	//================ GPU computing (with data transfering) ================
	//=======================================================================
	// 1. merge the two in GPU;
	// 2. transmit back, and then merge with the one in CPU;
	// 3. calculate the error, and load into GPU memory
	gpu_merge_list<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_gene_rpkm_exp_cellenv, d_gene_rpkm_exp_batch);
	float * temp_list = (float *)malloc(num_gene*sizeof(float));
	checkCudaErrors(cudaMemcpy(temp_list, d_gene_rpkm_exp_cellenv, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	for(long int i=0; i<num_gene; i++)
	{
		expr_con_pointer[i] = expr_con_pointer_cis[i] + temp_list[i];
	}
	free(temp_list);
	// error is the thing actually needed
	float * error_list = (float *)calloc(num_gene, sizeof(float));
	for(long int i=0; i<num_gene; i++)
	{
		error_list[i] = expr_con_pointer[i] - (*expr_list_pointer)[i];
	}
    checkCudaErrors(cudaMemcpy( d_error_list, error_list, num_gene*sizeof(float), cudaMemcpyHostToDevice));
    */


	//============== timing starts ================
	gettimeofday(&time_start, NULL);

	//==========================================================================
	//================ GPU computing (without data transfering) ================
	//==========================================================================
	// 1. merge the three in GPU;
	// 2. calculate the error in GPU
	gpu_merge_list_3<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_gene_rpkm_exp_cis, d_gene_rpkm_exp_cellenv, d_gene_rpkm_exp_batch);
	gpu_error_cal<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_error_list, d_gene_rpkm_exp_cis, d_expr);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("merge: Time used totally is %f seconds.\n", diff);




	/*
	// DEBUG
	sprintf(filename, "%s", "../result_tempdata/var_expr_exp.txt");
	para_temp_save_var(expr_con_pointer, num_gene, filename);
	// DEBUG: check the error
	sprintf(filename, "%s", "../result_tempdata/var_expr_error.txt");
	para_temp_save_var(error_list, num_gene, filename);
	*/




	//========================================================================
	//========================================================================
	// step#2: back-propogation (cis-;  cell env; batch)
	//========================================================================
	//========================================================================
	// *********************** [part1] cis- ************************
	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	backward_error_prop_direct_imcomp(matrix_imcomp_para_dev_cis_gene, error_list, dosage_list_pointer);
	*/

	//============== timing starts ================
	gettimeofday(&time_start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	gpu_backprop_cis<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis, num_gene, d_list_para_dev_cis_gene[etissue_index], d_error_list, d_snp,\
						d_cis_para_start, d_cis_para_amount, d_cis_snp_start, d_cis_para_index1);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("cis: Time used totally is %f seconds.\n", diff);




	// ***************** [part2] cell env relevant parameters *****************
	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	//// from cell env to genes
	backward_error_prop_last_layer(matrix_para_dev_cellenv_gene, error_list, cellenv_con_pointer);

	//// from snp to cell env
	backward_error_prop_inter_layer_1(error_list, cube_para_cellenv_gene[etissue_index], matrix_para_dev_snp_cellenv, cellenv_con_pointer, dosage_list_pointer);
	*/

	//============== timing starts ================
	gettimeofday(&time_start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	//==== from cellenv to gene expression (last layer)
	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
	float * d_para_dev_cellenv_gene = d_list_para_dev_cellenv_gene[etissue_index];
	gpu_backprop_last_layer<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_para_dev_cellenv_gene, d_error_list, d_cellenv_hidden_var);

	//==== from SNP to cellenv (first layer)
	// 1. calculate the temp list (saved into hidden variables)
	// 2. twist it with inverse neuralnet ac func
	// 3. use it as input for previous layer
	//
	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
	d_para_cellenv_gene = d_list_para_cellenv_gene[etissue_index];
	gpu_backprop_error_prop<<<( num_cellenv + 255 )/256 , 256 >>>( dimension1, dimension2, d_para_cellenv_gene, d_error_list, d_cellenv_hidden_var);
	//
	gpu_neuralnet_ac_func_dev<<<( num_cellenv + 255 )/256 , 256 >>>( num_cellenv, d_cellenv_hidden_var);
	//
	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
	gpu_backprop_last_layer<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_para_dev_snp_cellenv, d_cellenv_hidden_var, d_snp);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("cellenv: Time used totally is %f seconds.\n", diff);





	// ********************* [part3] linear or non-linear batches *********************
	/*
	//===============================================
	//================ CPU computing ================
	//===============================================
	//// from hidden batch to genes
	backward_error_prop_last_layer(matrix_para_dev_batch_hidden_gene, error_list, batch_hidden_con_pointer);

	// from original batch to hidden batch
	backward_error_prop_inter_layer_2(error_list, matrix_para_batch_hidden_gene, matrix_para_dev_batch_batch_hidden, batch_hidden_con_pointer, batch_list_pointer);
	*/

	//============== timing starts ================
	gettimeofday(&time_start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	//==== from hidden batch to genes (last layer)
	dimension1 = matrix_para_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_batch_hidden_gene.get_dimension2();
	gpu_backprop_last_layer<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_para_dev_batch_hidden_gene, d_error_list, d_batch_hidden_var);
	//==== from original batch to hidden batch
	// 1. calculate the temp list (saved into hidden variables)
	// 2. twist it with inverse neuralnet ac func
	// 3. use it as input for previous layer
	//
	dimension1 = matrix_para_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_batch_hidden_gene.get_dimension2();
	gpu_backprop_error_prop<<<( num_batch_hidden + 255 )/256 , 256 >>>( dimension1, dimension2, d_para_batch_hidden_gene, d_error_list, d_batch_hidden_var);
	//
	gpu_neuralnet_ac_func_dev<<<( num_batch_hidden + 255 )/256 , 256 >>>( num_batch_hidden, d_batch_hidden_var);
	//
	dimension1 = matrix_para_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_batch_batch_hidden.get_dimension2();
	gpu_backprop_last_layer<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_para_dev_batch_batch_hidden, d_batch_hidden_var, d_batch_var);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("batch: Time used totally is %f seconds.\n", diff);





	//free(error_list);

	return;
}







void regularization(int etissue_index)
{
	cout << "[@@] entering the regularization routine..." << endl;

	// there are several classes of prior knowledge that we need to consider
	// 1. sparsity of cis- regulation, accompanied by ridge regression, achieved by elastic-net tuned by the prior number, and the distance prior
	// 2. sparsity (LASSO) for the coefficients from cell env to expression (with the assumption that one gene is only affected by several handful cell env)
	// 3.1.[TODO] hierarchical regularization tuned by the learned tissue hierarchy
	// 3.2.[TODO] or we can simply use group LASSO to encourage the tissue consistency
	// 4. penalize the batch variables hashly (from batch variables to batch_hidden, and from batch_hidden to genes)
	cout << "adding the regularization items to the derivatives..." << endl;

	//===================================== part#0 =====================================
	// initialize some learning parameters

	// define the sigma here that may be used by L1 regularization
	float sigma = 0.0001;

	// regularization strength lambda:
	// path#1: add the lambda_{LASSO} and lambda_{ridge} for the cis- regularization
	// path#2: add the lambda for cellenv-gene regularization
	// path#3: and the lambda for batch-batch_hidden and batch_hidden-gene

	// TODO: DEBUG, see below
	//float lambda_lasso = 1.0;
	//float lambda_ridge = 1.0;
	//float lambda_snp_cellenv = 1.0;
	//float lambda_cellenv_gene = 1.0;
	//float lambda_batch_batch_hidden = 1.0;
	//float lambda_batch_hidden_gene = 1.0;


	//========================================================================================================
	// DEBUG
	// Feb.12: (let's test different lambda for one path -- batch path)
	// different lambda strength (listed below) x different batch iterations (200, 1000)
	// Feb.14: it seems the group#2 and group#3 are good

	// group#1
	//lambda = 0.1;
	// float lambda_lasso = 0.1;
	// float lambda_ridge = 0.1;
	// float lambda_snp_cellenv = 0.1;
	// float lambda_cellenv_gene = 0.1;
	// float lambda_batch_batch_hidden = 0.1;
	// float lambda_batch_hidden_gene = 0.1;

	// group#2
	//lambda = 0.001;
	float lambda_lasso = 0.001;
	float lambda_ridge = 0.001;
	float lambda_snp_cellenv = 0.001;
	float lambda_cellenv_gene = 0.001;
	float lambda_batch_batch_hidden = 0.001;
	float lambda_batch_hidden_gene = 0.001;

	// group#3
	// float lambda_lasso = 0.00001;
	// float lambda_ridge = 0.00001;
	// float lambda_snp_cellenv = 0.00001;
	// float lambda_cellenv_gene = 0.00001;
	// float lambda_batch_batch_hidden = 0.00001;
	// float lambda_batch_hidden_gene = 0.00001;

	// group#4
	// float lambda_lasso = 0.0000001;
	// float lambda_ridge = 0.0000001;
	// float lambda_snp_cellenv = 0.0000001;
	// float lambda_cellenv_gene = 0.0000001;
	// float lambda_batch_batch_hidden = 0.0000001;
	// float lambda_batch_hidden_gene = 0.0000001;



	// group#11
	// float lambda_lasso = 10;
	// float lambda_ridge = 10;
	// float lambda_snp_cellenv = 10;
	// float lambda_cellenv_gene = 10;
	// float lambda_batch_batch_hidden = 10;
	// float lambda_batch_hidden_gene = 10;

	// group#12
	// float lambda_lasso = 1000;
	// float lambda_ridge = 1000;
	// float lambda_snp_cellenv = 1000;
	// float lambda_cellenv_gene = 1000;
	// float lambda_batch_batch_hidden = 1000;
	// float lambda_batch_hidden_gene = 1000;

	// group#13
	// float lambda_lasso = 100000;
	// float lambda_ridge = 100000;
	// float lambda_snp_cellenv = 100000;
	// float lambda_cellenv_gene = 100000;
	// float lambda_batch_batch_hidden = 100000;
	// float lambda_batch_hidden_gene = 100000;


	//========================================================================================================
	/*
	//============== timing starts ================
    struct timeval time_start;
    struct timeval time_end;
    double diff;
    gettimeofday(&time_start, NULL);

	//=======================================
	//============ CPU Computing ============
	//=======================================
	//===================================== part#1 =====================================
	// 1. sparsity of cis- regulation, accompanied by ridge regression, achieved by elastic-net tuned by the prior number, (and the distance prior)
	// TODO: not yet integrated the distance prior information
	para_penalty_cis(cube_para_cis_gene[etissue_index], cube_para_dev_cis_gene[etissue_index], prior_tissue_vector[etissue_index], lambda_lasso, lambda_ridge, sigma);

	//===================================== part#2 =====================================
	// 2.1. snp to cellenv
	para_penalty_lasso_approx(matrix_para_snp_cellenv, matrix_para_dev_snp_cellenv, lambda_snp_cellenv, sigma);
	// 2.2. sparsity (LASSO) for the coefficients from cell env to expression (with the assumption that one gene is only affected by several handful cell env)
	para_penalty_lasso_approx(cube_para_cellenv_gene[etissue_index], cube_para_dev_cellenv_gene[etissue_index], lambda_cellenv_gene, sigma);

	//===================================== part#3 =====================================
	// 3.1. tissue hierarchy regularization;
	// 3.2. or we can simply use group LASSO to encourage the tissue consistency;
	// TODO: test later on
	//
	//
	//

	//===================================== part#4 =====================================
	// 4. penalize the batch variables hashly
	// 4.1. from batch to batch_hidden:
	para_penalty_lasso_approx(matrix_para_batch_batch_hidden, matrix_para_dev_batch_batch_hidden, lambda_batch_batch_hidden, sigma);
	// 4.2. from batch_hidden to gene:
	para_penalty_lasso_approx(matrix_para_batch_hidden_gene, matrix_para_dev_batch_hidden_gene, lambda_batch_hidden_gene, sigma);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);
	*/




	//=======================================
	//============ GPU Computing ============
	//=======================================
	gpu_penalty_cis<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis , d_list_para_cis_gene[etissue_index], d_list_para_dev_cis_gene[etissue_index], lambda_lasso, lambda_ridge, sigma);

	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	long int dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	gpu_penalty<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_snp_cellenv, d_para_dev_snp_cellenv, lambda_snp_cellenv, sigma);

	//vector<Matrix> cube_para_dev_cellenv_gene --> vector<float *> d_list_para_dev_cellenv_gene
	dimension1 = cube_para_dev_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[etissue_index].get_dimension2();
	float * d_para_cellenv_gene = d_list_para_cellenv_gene[etissue_index];
	float * d_para_dev_cellenv_gene = d_list_para_dev_cellenv_gene[etissue_index];
	gpu_penalty<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_cellenv_gene, d_para_dev_cellenv_gene, lambda_cellenv_gene, sigma);

	//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
	gpu_penalty<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_batch_batch_hidden, d_para_dev_batch_batch_hidden, lambda_batch_batch_hidden, sigma);

	//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
	gpu_penalty<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_batch_hidden_gene, d_para_dev_batch_hidden_gene, lambda_batch_hidden_gene, sigma);







	cout << "[@@] leaving the regularization routine..." << endl;
}





// for all parameters in our scope, we do p = p - rate_learner * dp (we have all the components in the right hand, as followed)
void gradient_descent(int etissue_index)
{
	cout << "[@@] entering the gradient descent..." << endl;







	/*
	//============== timing starts ================
    struct timeval time_start;
    struct timeval time_end;
    double diff;
    gettimeofday(&time_start, NULL);

	//=======================================
	//============ CPU Computing ============
	//=======================================
	//============================================ pathway#1 ================================================
	//====================== cube_para_cis_gene ==========================
	para_gradient_descent_cis(cube_para_cis_gene[etissue_index], cube_para_dev_cis_gene[etissue_index], rate_learner);

	//============================================ pathway#2 ================================================
	//====================== matrix_para_snp_cellenv ==========================
	para_gradient_descent(matrix_para_snp_cellenv, matrix_para_dev_snp_cellenv, rate_learner);

	//====================== cube_para_cellenv_gene ==========================
	para_gradient_descent(cube_para_cellenv_gene[etissue_index], cube_para_dev_cellenv_gene[etissue_index], rate_learner);

	//============================================ pathway#3 ================================================
	//====================== matrix_para_batch_batch_hidden ==========================
	para_gradient_descent(matrix_para_batch_batch_hidden, matrix_para_dev_batch_batch_hidden, rate_learner);

	//====================== matrix_para_batch_hidden_gene ==========================
	para_gradient_descent(matrix_para_batch_hidden_gene, matrix_para_dev_batch_hidden_gene, rate_learner);

	//============== timing ends ================
	gettimeofday(&time_end, NULL);
	diff = (double)(time_end.tv_sec-time_start.tv_sec) + (double)(time_end.tv_usec-time_start.tv_usec)/1000000;
	printf("Time used totally is %f seconds.\n", diff);
	*/



	//===============================================
	//================ GPU computing ================
	//===============================================
	gpu_gd<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis , d_list_para_cis_gene[etissue_index], d_list_para_dev_cis_gene[etissue_index], rate_learner);

	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	long int dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	gpu_gd<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_snp_cellenv, d_para_dev_snp_cellenv, rate_learner);

	//vector<Matrix> cube_para_dev_cellenv_gene --> vector<float *> d_list_para_dev_cellenv_gene
	dimension1 = cube_para_dev_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_dev_cellenv_gene[etissue_index].get_dimension2();
	float * d_para_cellenv_gene = d_list_para_cellenv_gene[etissue_index];
	float * d_para_dev_cellenv_gene = d_list_para_dev_cellenv_gene[etissue_index];
	gpu_gd<<< ( (dimension1*dimension2) + 255 )/256 , 256 >>>( (dimension1*dimension2) , d_para_cellenv_gene, d_para_dev_cellenv_gene, rate_learner);

	//Matrix matrix_para_dev_batch_batch_hidden --> float * d_para_dev_batch_batch_hidden
	dimension1 = matrix_para_dev_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_dev_batch_batch_hidden.get_dimension2();
	gpu_gd<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_batch_batch_hidden, d_para_dev_batch_batch_hidden, rate_learner);

	//Matrix matrix_para_dev_batch_hidden_gene --> float * d_para_dev_batch_hidden_gene
	dimension1 = matrix_para_dev_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_dev_batch_hidden_gene.get_dimension2();
	gpu_gd<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_batch_hidden_gene, d_para_dev_batch_hidden_gene, rate_learner);






	cout << "[@@] leaving the gradient descent..." << endl;
}















// calculate the loglikelihood for all the samples in one tissue
float cal_loglike(string etissue)
{
	cout << "now calculating the log-likelihood..." << endl;

	float loglike = 0;
	int etissue_index = etissue_index_map[etissue];

	for(int pos = 0; pos < esample_tissue_rep[etissue].size(); pos++)
	{
		loglike += forward_loglike_testerror(0, etissue, pos, &snp_dosage_list, gene_rpkm_exp, cellenv_hidden_var, batch_var, batch_hidden_var);	// indicator=0 --> loglike; indicator=1 --> testerror
	}

	return loglike;
}




// forward process, accumulated the errors
float forward_loglike_testerror(int indicator, string etissue, int pos, array<float *, NUM_CHR> * dosage_list_pointer, float * expr_con_pointer, float * cellenv_con_pointer, float * batch_list_pointer, float * batch_hidden_con_pointer)
{
	float result = 0;
	int etissue_index = etissue_index_map[etissue];


	//========================================================================
	// prepare all the containers, and the input variables
	//========================================================================
	//========================================================================
	//=================================================================================================================
	//******************************************* loglike or testerror ************************************************
	//=================================================================================================================
	string esample;
	if(indicator == 0)	// training set, loglike
	{
		esample = esample_tissue_rep[etissue][pos];
	}
	else 				// testing set, testerror
	{
		esample = esample_tissue_rep_test[etissue][pos];
	}
	string individual = sample_to_individual(esample);
	cout << "loglike: current sample #" << pos+1 << ": " << esample << endl;


	// make it compatible with the old code
	//array<float *, NUM_CHR> * dosage_list_pointer = &snp_dosage_list;
	//=================================================================================================================
	//******************************************* loglike or testerror ************************************************
	//=================================================================================================================
	vector<float> * expr_list_pointer;
	if(indicator == 0)	// training set, loglike
	{
		expr_list_pointer = &eQTL_tissue_rep[etissue][esample];
	}
	else 				// testing set, testerror
	{
		expr_list_pointer = &eQTL_tissue_rep_test[etissue][esample];
	}
	//float * expr_con_pointer = gene_rpkm_exp;
	//float * cellenv_con_pointer = cellenv_hidden_var;
	//float * batch_list_pointer = batch_var;
	//float * batch_hidden_con_pointer = batch_hidden_var;



	//=================================================== init ============================================================
	// get the: 0. esample and individual; 1. genotype; 2. expression data; 3. batch variables
	// to: 1. forward_backward propagation;
	// genotype dosage data
	//cout << "getting the dosage data for individual #" << individual << endl;
	snp_dosage_load(dosage_list_pointer, individual);  // snp dosage data for one individual across all chromosomes
	// expression rpkm data: eQTL_tissue_rep[etissue][esample]
	//cout << "we have this amount of genes expressed in this individual:" << eQTL_tissue_rep[etissue][esample].size() << endl;
	// and the batch variable for this individual and this sample
	int num_batch_individual = batch_individual[individual].size();
	int index = 0;
	for(int i=0; i<num_batch_individual; i++)
	{
		float value = batch_individual[individual][i];
		batch_list_pointer[index] = value;
		index++;
	}
	int num_batch_sample = batch_sample[esample].size();
	for(int i=0; i<num_batch_sample; i++)
	{
		float value = batch_sample[esample][i];
		batch_list_pointer[index] = value;
		index++;
	}


	//========================================================================
	// forward-propogation (cis-; cell env; batch)
	//========================================================================
	//========================================================================

	// ****************************** [part1] cis- *********************************
	// for cis-, two issues:
	// 1. if this is a XYMT gene, we don't have signal from it's cis- SNPs (not consider currently);
	// 2. we use (gene_cis_index[gene].second - gene_cis_index[gene].first + 1) as the length of the cis- parameter array
	float * expr_con_pointer_cis = (float *)calloc( num_gene, sizeof(float) );
	multi_array_matrix_imcomp(dosage_list_pointer, cube_para_cis_gene[etissue_index], expr_con_pointer_cis);


	// ********************* [part2] cell env relevant parameters *********************
	// from snp to cell env variables
	float * expr_con_pointer_cellenv = (float *)calloc( num_gene, sizeof(float) );
	multi_array_list_matrix(dosage_list_pointer, matrix_para_snp_cellenv, cellenv_con_pointer);

	//$$$$$$$$$$$ perform the activation function here (logistic or something else) $$$$$$$$$$$$
	neuralnet_ac_func(cellenv_con_pointer, num_cellenv);

	// from cell env variables to genes
	multi_array_matrix(cellenv_con_pointer, cube_para_cellenv_gene[etissue_index], expr_con_pointer_cellenv);


	// ********************* [part3] linear or non-linear batches *********************
	float * expr_con_pointer_batch = (float *)calloc( num_gene, sizeof(float) );
	// from original batch to hidden batch
	multi_array_matrix(batch_list_pointer, matrix_para_batch_batch_hidden, batch_hidden_con_pointer);

	//$$$$$$$$$$$ perform the activation function here (logistic or something else) $$$$$$$$$$$$
	neuralnet_ac_func(batch_hidden_con_pointer, num_batch_hidden);

	// from hidden batch to genes
	multi_array_matrix(batch_hidden_con_pointer, matrix_para_batch_hidden_gene, expr_con_pointer_batch);



	// ********************* [end] merge the signal from three pathways here, to expr_con_pointer *********************
	for(long int i=0; i<num_gene; i++)
	{
		expr_con_pointer[i] = expr_con_pointer_cis[i] + expr_con_pointer_cellenv[i] + expr_con_pointer_batch[i];
	}
	free(expr_con_pointer_cis);
	free(expr_con_pointer_cellenv);
	free(expr_con_pointer_batch);



	// error is the thing actually needed
	//=================================================================================================================
	//******************************************* loglike or testerror ************************************************
	//=================================================================================================================
	if(indicator == 0)		// training set, loglike
	{
		float loglike = 0;
		for(long int i=0; i<num_gene; i++)
		{
			float error = expr_con_pointer[i] - (*expr_list_pointer)[i];
			// save the loglike
			loglike -= pow(error, 2.0);
		}
		result = loglike;
	}
	else 					// testing error, testerror
	{
		float testerror = 0;
		for(long int i=0; i<num_gene; i++)
		{
			float error = abs(expr_con_pointer[i] - (*expr_list_pointer)[i]);
			testerror += error;
		}
		result = testerror;
	}


	return result;
}



// testing error for one tissue
// type: MAE or AE (mean absolute error or absolute error)
float cal_testerror(string etissue)
{
	float testerror = 0;

	cout << "now calculating the testing error (for the current tissue)..." << endl;

	int etissue_index = etissue_index_map[etissue];

	for(int pos = 0; pos < esample_tissue_rep_test[etissue].size(); pos++)
	{
		testerror += forward_loglike_testerror(1, etissue, pos, &snp_dosage_list, gene_rpkm_exp, cellenv_hidden_var, batch_var, batch_hidden_var);	// indicator=0 --> loglike; indicator=1 --> testerror
	}


	return testerror;
}


