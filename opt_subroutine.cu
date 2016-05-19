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
#include "opt_subroutine_gpu_mm.cuh"

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
	//gpu_addone<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_snp_cellenv);

	// DEBUG: test the maximum possible size of a grid
	//gpu_addone<<< 65535 , 256 >>>( (dimension1*dimension2) , d_para_snp_cellenv);
	gpu_addone<<<( (dimension1*dimension2)+1023)/1024 , 1024 >>>( (dimension1*dimension2) , d_para_snp_cellenv);
	// NOTE: be careful on the maximum possible blocks


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



	// temp variables for GPU timing:
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);
    float msecTotal = 0.0f;




	//============== timing starts ================
	cudaEventRecord(start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	// clean GPU memory for para_dev (as the forward_backward function is additive)
	gpu_clean<<<( num_para_cis+255 )/256 , 256 >>>( num_para_cis , d_list_para_dev_cis_gene[etissue_index]);

	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	//gpu_clean<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_dev_snp_cellenv);
	//============================
	// NOTE: grid size
	//============================
	gpu_clean<<<( (dimension1*dimension2)+1023)/1024 , 1024 >>>( (dimension1*dimension2) , d_para_dev_snp_cellenv);

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
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("clean: Time used totally is %.3f msec.\n", msecTotal);





	//****************************** enter the mini-batch ***********************************
	cout << "we are entering a new mini-batch..." << endl;
	for(int count=0; count<batch_size; count++)
	{




		// temp variables for GPU timing:
		// Allocate CUDA events that we'll use for timing
		cudaEvent_t start;
		cudaEventCreate(&start);
		cudaEvent_t stop;
		cudaEventCreate(&stop);
	    float msecTotal = 0.0f;




		//==========================================================================================================
		//========================================== CPU data preparation ==========================================
		//==========================================================================================================
		//============== timing starts ================
	    // Record the start event
		cudaEventRecord(start, NULL);

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
	    // Record the stop event
		cudaEventRecord(stop, NULL);
	    // Wait for the stop event to complete
		cudaEventSynchronize(stop);
		// Timing
		cudaEventElapsedTime(&msecTotal, start, stop);
	    // Compute and print the performance
		printf("pre- Time used totally is %.3f msec.\n", msecTotal);




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
		//========================================== GPU data preparation ==========================================
		//==========================================================================================================
		//============== timing starts ================
		// Record the start event
		cudaEventRecord(start, NULL);

		// TODO GPU: load everything into GPU memory, and make them addressable
		// NOTE: CPU should have all the data, and GPU has also another copy of these data
		//===============================================
		//================ GPU computing ================
		//===============================================
		// //==== int * d_etissue_index_p
		// checkCudaErrors(cudaMemcpy( d_etissue_index_p, &etissue_index, 1*sizeof(int), cudaMemcpyHostToDevice));

    	/*
    	//===================
		//==== float * d_snp
    	//===================
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
	    // Record the stop event
		cudaEventRecord(stop, NULL);
	    // Wait for the stop event to complete
		cudaEventSynchronize(stop);
		// Timing
		cudaEventElapsedTime(&msecTotal, start, stop);
	    // Compute and print the performance
		printf("dosage GPU: Time used totally is %.3f msec.\n", msecTotal);




		//============== timing starts ================
		// Record the start event
		cudaEventRecord(start, NULL);

		/*
		//=========================
		//==== float * d_batch_var
		//=========================
		checkCudaErrors(cudaMemcpy( d_batch_var, batch_var, num_batch*sizeof(float), cudaMemcpyHostToDevice));
		*/

		int batch_index = d_batch_index_map[esample];
		d_batch_var = d_batch_list[batch_index];

		//============== timing ends ================
	    // Record the stop event
		cudaEventRecord(stop, NULL);
	    // Wait for the stop event to complete
		cudaEventSynchronize(stop);
		// Timing
		cudaEventElapsedTime(&msecTotal, start, stop);
	    // Compute and print the performance
		printf("batch GPU: Time used totally is %.3f msec.\n", msecTotal);






		//============== timing starts ================
		// Record the start event
		cudaEventRecord(start, NULL);

    	/*
		//====================
		//==== float * d_expr
		//====================
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
	    // Record the stop event
		cudaEventRecord(stop, NULL);
	    // Wait for the stop event to complete
		cudaEventSynchronize(stop);
		// Timing
		cudaEventElapsedTime(&msecTotal, start, stop);
	    // Compute and print the performance
		printf("expr: Time used totally is %.3f msec.\n", msecTotal);





		//============== timing starts ================
		// Record the start event
		cudaEventRecord(start, NULL);

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
	    // Record the stop event
		cudaEventRecord(stop, NULL);
	    // Wait for the stop event to complete
		cudaEventSynchronize(stop);
		// Timing
		cudaEventElapsedTime(&msecTotal, start, stop);
	    // Compute and print the performance
		printf("(forward_backward) Time used totally is %.3f msec.\n", msecTotal);

		// for extracting this item from log file
		printf("###### %.3f\n", msecTotal);






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





	/*
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
	*/






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
	// Record the start event
	cudaEventRecord(start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	float factor = 1.0 / batch_size;

	gpu_scale<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis , factor, d_list_para_dev_cis_gene[etissue_index]);

	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	//gpu_scale<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , factor, d_para_dev_snp_cellenv);
	//============================
	// NOTE: grid size
	//============================
	gpu_scale<<<( (dimension1*dimension2)+1023)/1024 , 1024 >>>( (dimension1*dimension2) , factor, d_para_dev_snp_cellenv);

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
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("aggre: Time used totally is %.3f msec.\n", msecTotal);







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






	//===================================== Regularization in Regression =====================================
	// NOTE: not yet tested, but this is too straightforward to test
	regularization(etissue_index);




	//=========================================== Gradient Descent ===========================================
	// NOTE: this routine is tested to be correct (Apr.25)
	gradient_descent(etissue_index);





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
	char filename[100];




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




	long int dimension1, dimension2;
	long int pos_start;
	int BLOCK_SIZE;
	dim3 threads;
	dim3 grid;

	//============== timing ================
	struct timeval time_start;
	struct timeval time_end;
	double diff;

	// temp variables for GPU timing:
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	float msecTotal = 0.0f;








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
    // Record the start event
	cudaEventRecord(start, NULL);

	//========================================================
	//================ GPU computing (naive1) ================
	//========================================================
	//== multiply the input to the matrix
	gpu_matrix_mul_cis_mul<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis, num_gene, d_snp, d_list_para_cis_gene[etissue_index], d_temp_cis_gene,\
							d_cis_para_start, d_cis_para_amount, d_cis_snp_start, d_cis_para_index1);

	// BEBUG
	/*
	// DEBUG: check d_temp_cis_gene
	cout << num_para_cis << endl;

	// transfer d_temp_cis_gene back
	float * temp_list = (float *)malloc(num_para_cis*sizeof(float));
	checkCudaErrors(cudaMemcpy(temp_list, d_temp_cis_gene, num_para_cis*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/temp_cis_GPU.txt");
	para_temp_save_var(temp_list, num_para_cis, filename);
	free(temp_list);

	// copy back the parameters also --> Matrix_imcomplete transfer is correct
	//vector<Matrix_imcomp> cube_para_cis_gene--> vector<float *> d_list_para_cis_gene;
	dimension1 = cube_para_cis_gene[etissue_index].get_dimension1();
	pos_start = 0;
	for(long int i=0; i<dimension1; i++)
	{
		long int amount = cube_para_cis_gene[etissue_index].get_dimension2(i);
		float * x = cube_para_cis_gene[etissue_index].get_list(i);
		checkCudaErrors(cudaMemcpy(x, (d_list_para_cis_gene[etissue_index] + pos_start), amount*sizeof(float), cudaMemcpyDeviceToHost));
		pos_start += amount;
	}

	// calculate d_temp_cis_gene locally
	temp_list = (float *)malloc(num_para_cis*sizeof(float));
	long int count = 0;
	Matrix_imcomp matrix_imcomp_para = cube_para_cis_gene[etissue_index];
	dimension1 = matrix_imcomp_para.get_dimension1();
	for(long int i=0; i<dimension1; i++)
	{
		int chr = matrix_imcomp_para.get_chr(i);
		long int start = matrix_imcomp_para.get_sst(i);
		long int dimension2 = matrix_imcomp_para.get_dimension2(i);

		for(long int j=0; j<dimension2; j++)
		{
			if(j == dimension2 - 1)
			{
				float par = matrix_imcomp_para.get(i, j);
				temp_list[count] = 1 * par;			// the last one in the parameter list is for the intercept term
			}
			else
			{
				long int pos = start + j;
				float var = (*dosage_list_pointer)[chr-1][pos];
				float par = matrix_imcomp_para.get(i, j);
				temp_list[count] = var * par;
			}
			count += 1;
		}
	}
	sprintf(filename, "%s", "../result_tempdata/temp_cis_normal.txt");
	para_temp_save_var(temp_list, num_para_cis, filename);
	free(temp_list);
	*/


	//== sum matrix
	gpu_matrix_mul_cis_add<<<(num_gene + 255) / 256 , 256 >>>( num_gene, d_temp_cis_gene, d_gene_rpkm_exp_cis,\
							d_cis_para_start, d_cis_para_amount, d_cis_snp_start);

	/*
	// DEBUG
	float * expr_con_pointer_cis = (float *)calloc( num_gene, sizeof(float) );
	checkCudaErrors(cudaMemcpy(expr_con_pointer_cis, d_gene_rpkm_exp_cis, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_expr_cis.txt");
	para_temp_save_var(expr_con_pointer_cis, num_gene, filename);
	free(expr_con_pointer_cis);
	*/

	// DEBUG
	/*
	// direct computation
	expr_con_pointer_cis = (float *)calloc( num_gene, sizeof(float) );
	multi_array_matrix_imcomp(dosage_list_pointer, cube_para_cis_gene[etissue_index], expr_con_pointer_cis);
	sprintf(filename, "%s", "../result_tempdata/var_expr_cis_normal1.txt");
	para_temp_save_var(expr_con_pointer_cis, num_gene, filename);
	free(expr_con_pointer_cis);

	// GPU transfer-back computation
	dimension1 = cube_para_cis_gene[etissue_index].get_dimension1();
	pos_start = 0;
	for(long int j=0; j<dimension1; j++)
	{
		long int amount = cube_para_cis_gene[etissue_index].get_dimension2(j);
		float * x = cube_para_cis_gene[etissue_index].get_list(j);
		checkCudaErrors(cudaMemcpy(x, (d_list_para_cis_gene[etissue_index] + pos_start), amount*sizeof(float), cudaMemcpyDeviceToHost));
		pos_start += amount;
	}

	expr_con_pointer_cis = (float *)calloc( num_gene, sizeof(float) );
	multi_array_matrix_imcomp(dosage_list_pointer, cube_para_cis_gene[etissue_index], expr_con_pointer_cis);

	sprintf(filename, "%s", "../result_tempdata/var_expr_cis_normal2.txt");
	para_temp_save_var(expr_con_pointer_cis, num_gene, filename);
	free(expr_con_pointer_cis);
	*/


	/*
	// DEBUG
	gpu_clean<<<(num_gene + 255) / 256 , 256 >>>( num_gene, d_gene_rpkm_exp_cis);

	// DEBUG
	float * expr_con_pointer_cis = (float *)calloc( num_gene, sizeof(float) );
	checkCudaErrors(cudaMemcpy(expr_con_pointer_cis, d_gene_rpkm_exp_cis, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_expr_cis.txt");
	para_temp_save_var(expr_con_pointer_cis, num_gene, filename);
	free(expr_con_pointer_cis);
	*/

	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("cis: Time used totally is %.3f msec.\n", msecTotal);






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




	/*
	cout << "test ..." << endl;
	//========================================================================
	// May.17: take a middle, do CPU computation, and transmit results to GPU
	//========================================================================
	//==== transmit data and para back from GPU
	//matrix_para_snp_cellenv --> float * d_para_snp_cellenv
	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_snp_cellenv + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }
	//vector<Matrix> cube_para_cellenv_gene --> <float *> d_list_para_cellenv_gene
	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
	for(long int i=0; i<dimension1; i++)
	{
		float * x = cube_para_cellenv_gene[etissue_index].get_list(i);
		long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_list_para_cellenv_gene[etissue_index] + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
	}

	//==== do CPU computation
	// from snp to cell env variables
	float * expr_con_pointer_cellenv = (float *)calloc( num_gene, sizeof(float) );
	multi_array_list_matrix(dosage_list_pointer, matrix_para_snp_cellenv, cellenv_con_pointer);
	//$$$$$$$$$$$ perform the activation function here (logistic or something else) $$$$$$$$$$$$
	neuralnet_ac_func(cellenv_con_pointer, num_cellenv);
	// from cell env variables to genes
	multi_array_matrix(cellenv_con_pointer, cube_para_cellenv_gene[etissue_index], expr_con_pointer_cellenv);

	//==== transmit results back to GPU
	checkCudaErrors(cudaMemcpy( d_gene_rpkm_exp_cellenv, expr_con_pointer_cellenv, num_gene*sizeof(float), cudaMemcpyHostToDevice));
	free(expr_con_pointer_cellenv);
	*/





	//============== timing starts ================
    // Record the start event
	cudaEventRecord(start, NULL);


	/*
	//===============================================
	//================ GPU computing ================ (Apr.27, has been debugged, for both MM and neural hidden layer)
	//===============================================
	//==== from SNP to cellenv
	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
	//== multiply the input to the matrix
	//============================
	// NOTE: grid size
	//============================
	//gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_snp, d_para_snp_cellenv, d_temp_snp_cellenv);
	gpu_matrix_mul_mul<<<( (dimension1*dimension2)+1023)/1024 , 1024 >>>( dimension1, dimension2, d_snp, d_para_snp_cellenv, d_temp_snp_cellenv);
	//== sum matrix
	gpu_matrix_mul_add<<<(dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_snp_cellenv, d_cellenv_hidden_var);
	*/




	/*
	// TESTING (the new MatrixMul methods)
	//==== from SNP to cellenv
	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
	BLOCK_SIZE = 32;
	long int WC = 1;
	long int HC = num_cellenv;
    // setup execution parameters
    //dim3 threads = dim3(1, BLOCK_SIZE);		// NOTE: the time increases from 60 msecs to 93.234 msecs after this
    											// NOTE: it seems that two-dimension structure makes things faster
	threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    //grid = dim3( ( WC + threads.x - 1 )/threads.x, ( HC + threads.y - 1 )/threads.y );
    // NOTE: I will make the testing dummy
	grid = dim3( ( WC + threads.x - 1 )/threads.x, ( HC + threads.y - 1 )/threads.y );

    // naive implementation
    matrixMul_naive<<< grid, threads >>>(d_cellenv_hidden_var, d_para_snp_cellenv, d_snp, dimension1, dimension2, 1);
    */





	// TESTING (the new MatrixMul methods, with shared memory scheme)
    // testing other algorithm (splitting the width of the huge matrix)
    // motivation:
    //	1. as each element in para matrix will only be loaded once, there is no need to preload
	//	2. the bottleneck is the parallism of computing the super long matrix
	// practice:
	//	1. the dosage will be used several times, thus preloading into block should be good
	//	2. each block (512) will deal with 400 lines, with length L
	//	3. L will depend on how much memory needed for preloading these dosage data, and the max number of blocks applicable
	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
	matrixMul_upgrade_multi<<< num_block, 512 >>>(dimension1, dimension2, L, num_block, d_cellenv_hidden_var_sub, d_para_snp_cellenv, d_snp);
	matrixMul_upgrade_sum<<< (num_cellenv+255)/256, 256 >>>(dimension1, num_block, d_cellenv_hidden_var_sub, d_cellenv_hidden_var);







	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("cellenv (huge MM): Time used totally is %.3f msec.\n", msecTotal);




	//============== timing starts ================
    // Record the start event
	cudaEventRecord(start, NULL);


	// The problem is in the above routine !!! --> I applied more blocks in a grid that's more than allowed


	// dimension1 = matrix_para_snp_cellenv.get_dimension1();
	// dimension2 = matrix_para_snp_cellenv.get_dimension2();
 //    for(long int i=0; i<dimension1; i++)
 //    {
 //    	float * x = matrix_para_snp_cellenv.get_list(i);
 //    	long int pos_start = i * dimension2;
	// 	checkCudaErrors(cudaMemcpy(x, (d_para_snp_cellenv + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
 //    }

 //    float * list_snp = (float *)malloc(num_snp*sizeof(float));
	// checkCudaErrors(cudaMemcpy(list_snp, d_snp, num_snp*sizeof(float), cudaMemcpyDeviceToHost));

 //    float * list_temp = (float *)malloc(num_cellenv*sizeof(float));
	// for(int i=0; i<dimension1; i++)
	// {
	// 	list_temp[i] = 0;
	// 	for(int j=0; j<dimension2; j++)
	// 	{
	// 		if(j == dimension2-1)		// the intercept
	// 		{
	// 			float para = matrix_para_snp_cellenv.get(i, j);
	// 			list_temp[i] += 1 * para;
	// 			break;
	// 		}

	// 		float para = matrix_para_snp_cellenv.get(i, j);
	// 		list_temp[i] += list_snp[j] * para;
	// 	}
	// }
	// sprintf(filename, "%s", "../result_tempdata/var_cellenv_before_normal.txt");
	// para_temp_save_var(list_temp, num_cellenv, filename);
	// free(list_snp);
	// free(list_temp);
		

	
	/*
	// DEBUG
	float * list_temp = (float *)calloc( num_cellenv, sizeof(float) );
	checkCudaErrors(cudaMemcpy(list_temp, d_cellenv_hidden_var, num_cellenv*sizeof(float), cudaMemcpyDeviceToHost));
	neuralnet_ac_func(list_temp, num_cellenv);
	sprintf(filename, "%s", "../result_tempdata/var_cellenv_after_normal.txt");
	para_temp_save_var(list_temp, num_cellenv, filename);
	free(list_temp);
	*/



	//==== neuralnet
	gpu_neuralnet_ac_func<<<(num_cellenv + 255) / 256 , 256 >>>( num_cellenv , d_cellenv_hidden_var);



	/*
	// DEBUG
	list_temp = (float *)calloc( num_cellenv, sizeof(float) );
	checkCudaErrors(cudaMemcpy(list_temp, d_cellenv_hidden_var, num_cellenv*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_cellenv_after_GPU.txt");
	para_temp_save_var(list_temp, num_cellenv, filename);
	free(list_temp);
	*/


	//==== from cellenv to gene expression
	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
	//== multiply the input to the matrix
	gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_cellenv_hidden_var, d_list_para_cellenv_gene[etissue_index], d_temp_cellenv_gene);
	//== sum matrix
	gpu_matrix_mul_add<<< (dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_cellenv_gene, d_gene_rpkm_exp_cellenv);


	/*
	// DEBUG
	float * expr_con_pointer_cellenv = (float *)calloc( num_gene, sizeof(float) );
	checkCudaErrors(cudaMemcpy(expr_con_pointer_cellenv, d_gene_rpkm_exp_cellenv, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_expr_cellenv.txt");
	para_temp_save_var(expr_con_pointer_cellenv, num_gene, filename);
	free(expr_con_pointer_cellenv);
	*/


	/*
	// DEBUG
	gpu_setnum<<<(num_gene + 255) / 256 , 256 >>>( num_gene, d_gene_rpkm_exp_cellenv, 1);

	// DEBUG
	float * expr_con_pointer_cellenv = (float *)calloc( num_gene, sizeof(float) );
	checkCudaErrors(cudaMemcpy(expr_con_pointer_cellenv, d_gene_rpkm_exp_cellenv, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_expr_cellenv.txt");
	para_temp_save_var(expr_con_pointer_cellenv, num_gene, filename);
	free(expr_con_pointer_cellenv);
	*/

	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("cellenv: Time used totally is %.3f msec.\n", msecTotal);
	












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
    // Record the start event
	cudaEventRecord(start, NULL);

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


	// probably I'll need to transfer every intermediate stuff back


	/*
	// calculate the var_batch_hidden and save it
	//Matrix matrix_para_batch_batch_hidden --> float * d_para_batch_batch_hidden
	dimension1 = matrix_para_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_batch_batch_hidden.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_batch_batch_hidden.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_batch_batch_hidden + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

    float * list_batch = (float *)malloc(num_batch*sizeof(float));
	checkCudaErrors(cudaMemcpy(list_batch, d_batch_var, num_batch*sizeof(float), cudaMemcpyDeviceToHost));

    float * list_temp = (float *)malloc(num_batch_hidden*sizeof(float));
	for(int i=0; i<dimension1; i++)
	{
		list_temp[i] = 0;
		for(int j=0; j<dimension2; j++)
		{
			if(j == dimension2-1)		// the intercept
			{
				float para = matrix_para_batch_batch_hidden.get(i, j);
				list_temp[i] += 1 * para;
				break;
			}

			float para = matrix_para_batch_batch_hidden.get(i, j);
			list_temp[i] += list_batch[j] * para;
		}
	}
	sprintf(filename, "%s", "../result_tempdata/var_batch_hidden_before_normal.txt");
	para_temp_save_var(list_temp, num_batch_hidden, filename);
	free(list_batch);
	free(list_temp);
	*/


	/*
	// DEBUG
	float * list_temp = (float *)calloc( num_batch_hidden, sizeof(float) );
	checkCudaErrors(cudaMemcpy(list_temp, d_batch_hidden_var, num_batch_hidden*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_batch_hidden_before_GPU.txt");
	para_temp_save_var(list_temp, num_batch_hidden, filename);
	free(list_temp);
	*/


	//==== neuralnet
	gpu_neuralnet_ac_func<<<(num_batch_hidden + 255) / 256 , 256 >>>( num_batch_hidden , d_batch_hidden_var);


	/*
	// DEBUG
	list_temp = (float *)calloc( num_batch_hidden, sizeof(float) );
	checkCudaErrors(cudaMemcpy(list_temp, d_batch_hidden_var, num_batch_hidden*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_batch_hidden_after.txt");
	para_temp_save_var(list_temp, num_batch_hidden, filename);
	free(list_temp);
	*/



	//==== from batch_hidden to gene
	dimension1 = matrix_para_batch_hidden_gene.get_dimension1();
	dimension2 = matrix_para_batch_hidden_gene.get_dimension2();
	//== multiply the input to the matrix
	gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_batch_hidden_var, d_para_batch_hidden_gene, d_temp_batch_hidden_gene);
	//== sum matrix
	gpu_matrix_mul_add<<< (dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_batch_hidden_gene, d_gene_rpkm_exp_batch);

	/*
	// DEBUG
	float * expr_con_pointer_batch = (float *)calloc( num_gene, sizeof(float) );
	checkCudaErrors(cudaMemcpy(expr_con_pointer_batch, d_gene_rpkm_exp_batch, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_expr_batch.txt");
	para_temp_save_var(expr_con_pointer_batch, num_gene, filename);
	free(expr_con_pointer_batch);
	*/

	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("batch: Time used totally is %.3f msec.\n", msecTotal);






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
    // Record the start event
	cudaEventRecord(start, NULL);

	//==========================================================================
	//================ GPU computing (without data transfering) ================
	//==========================================================================
	// 1. merge the three in GPU;
	// 2. calculate the error in GPU
	gpu_merge_list_3<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_gene_rpkm_exp_cis, d_gene_rpkm_exp_cellenv, d_gene_rpkm_exp_batch);
	gpu_error_cal<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_error_list, d_gene_rpkm_exp_cis, d_expr);

	/*
    // DEBUG
    float * error_list = (float *)malloc(num_gene*sizeof(float));
    float * expr_list = (float *)malloc(num_gene*sizeof(float));
	checkCudaErrors(cudaMemcpy(expr_con_pointer, d_gene_rpkm_exp_cis, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(error_list, d_error_list, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaMemcpy(expr_list, d_expr, num_gene*sizeof(float), cudaMemcpyDeviceToHost));
	sprintf(filename, "%s", "../result_tempdata/var_expr_exp.txt");
	para_temp_save_var(expr_con_pointer, num_gene, filename);
	sprintf(filename, "%s", "../result_tempdata/var_expr_real.txt");
	para_temp_save_var(expr_list, num_gene, filename);
	sprintf(filename, "%s", "../result_tempdata/var_expr_error.txt");
	para_temp_save_var(error_list, num_gene, filename);
	free(expr_list);
	*/

	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("merge: Time used totally is %.3f msec.\n", msecTotal);






	//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//###########################################################
	//===========================================================
	// NOTE: now the forward process is correct!! (Apr.27, 2016)
	//===========================================================
	//###########################################################
	//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
	//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$






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

	// //============== timing starts ================
    // Record the start event
	cudaEventRecord(start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	gpu_backprop_cis<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis, num_gene, d_list_para_dev_cis_gene[etissue_index], d_error_list, d_snp,\
						d_cis_para_start, d_cis_para_amount, d_cis_snp_start, d_cis_para_index1);

	//============== timing ends ================
	// Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("cis: Time used totally is %.3f msec.\n", msecTotal);







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


	/*
	///////// passway#2 CPU local test (passed)

	// I need the following:
    checkCudaErrors(cudaMemcpy( cellenv_con_pointer, d_cellenv_hidden_var, num_cellenv*sizeof(float), cudaMemcpyDeviceToHost));


	cout << "test ..." << endl;
	//========================================
	// test the local CPU computation version
	//========================================
	//==== transmit stuff back from GPU; do the following:
	// matrix_para_dev_cellenv_gene
	// error_list
	// //cellenv_con_pointer
	// cube_para_cellenv_gene[etissue_index]
	// matrix_para_dev_snp_cellenv
	// //cellenv_con_pointer
	// //dosage_list_pointer
	//vector<Matrix> cube_para_cellenv_gene --> <float *> d_list_para_cellenv_gene
	dimension1 = matrix_para_dev_cellenv_gene.get_dimension1();
	dimension2 = matrix_para_dev_cellenv_gene.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_cellenv_gene.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_list_para_dev_cellenv_gene[etissue_index] + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

	float * error_list = (float *)malloc(num_gene*sizeof(float));
	checkCudaErrors(cudaMemcpy(error_list, d_error_list, num_gene*sizeof(float), cudaMemcpyDeviceToHost));

	//vector<Matrix> cube_para_cellenv_gene --> <float *> d_list_para_cellenv_gene
	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = cube_para_cellenv_gene[etissue_index].get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_list_para_cellenv_gene[etissue_index] + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }

	//matrix_para_snp_cellenv --> float * d_para_snp_cellenv
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_snp_cellenv.get_list(i);
    	long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy(x, (d_para_dev_snp_cellenv + pos_start), dimension2*sizeof(float), cudaMemcpyDeviceToHost));
    }


	//==== CPU computing
	//// from cell env to genes
	backward_error_prop_last_layer(matrix_para_dev_cellenv_gene, error_list, cellenv_con_pointer);
	//// from snp to cell env
	backward_error_prop_inter_layer_1(error_list, cube_para_cellenv_gene[etissue_index], matrix_para_dev_snp_cellenv, cellenv_con_pointer, dosage_list_pointer);


	//==== transmit stuff again into GPU; do the following:
	// matrix_para_dev_cellenv_gene
	// matrix_para_dev_snp_cellenv
	//matrix_para_snp_cellenv --> float * d_para_snp_cellenv
	dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	for(long int i=0; i<dimension1; i++)
	{
		float * x = matrix_para_dev_snp_cellenv.get_list(i);
		long int pos_start = i * dimension2;
		checkCudaErrors(cudaMemcpy( (d_para_dev_snp_cellenv + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }

	//vector<Matrix> cube_para_cellenv_gene --> vector<float *> d_list_para_cellenv_gene
	dimension1 = matrix_para_dev_cellenv_gene.get_dimension1();
	dimension2 = matrix_para_dev_cellenv_gene.get_dimension2();
    for(long int i=0; i<dimension1; i++)
    {
    	float * x = matrix_para_dev_cellenv_gene.get_list(i);
    	long int pos_start = i * dimension2;
    	checkCudaErrors(cudaMemcpy( (d_list_para_dev_cellenv_gene[etissue_index] + pos_start), x, dimension2*sizeof(float), cudaMemcpyHostToDevice));
    }

	free(error_list);
	*/





	//============== timing starts ================
    // Record the start event
	cudaEventRecord(start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	//==== from cellenv to gene expression (last layer)
	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
	gpu_backprop_last_layer<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_list_para_dev_cellenv_gene[etissue_index], d_error_list, d_cellenv_hidden_var);


	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("cellenv (last layer): Time used totally is %.3f msec.\n", msecTotal);



	//============== timing starts ================
    // Record the start event
	cudaEventRecord(start, NULL);

	//==== from SNP to cellenv (first layer)
	// 1. calculate the temp list (saved into hidden variables)
	// 2. twist it with inverse neuralnet ac func
	// 3. use it as input for previous layer
	//
	dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
	dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
	gpu_backprop_error_prop<<<( num_cellenv + 255 )/256 , 256 >>>( dimension1, dimension2, d_list_para_cellenv_gene[etissue_index], d_error_list, d_cellenv_hidden_var_backerror);

	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("cellenv (error prop): Time used totally is %.3f msec.\n", msecTotal);



	//============== timing starts ================
    // Record the start event
	cudaEventRecord(start, NULL);

	// NOTE: this is not just logistic dev; it's combined with the propogated errors
	gpu_neuralnet_ac_func_dev_error<<<( num_cellenv + 255 )/256 , 256 >>>( num_cellenv, d_cellenv_hidden_var, d_cellenv_hidden_var_backerror);
	//

	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("cellenv (neural net): Time used totally is %.3f msec.\n", msecTotal);





	//============== timing starts ================
	// Record the start event
	cudaEventRecord(start, NULL);


	dimension1 = matrix_para_snp_cellenv.get_dimension1();
	dimension2 = matrix_para_snp_cellenv.get_dimension2();
	//gpu_backprop_last_layer<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_para_dev_snp_cellenv, d_cellenv_hidden_var, d_snp);
	//==============================================
	//NOTE: maximum number of blocks in a grid !!!
	//==============================================
	//gpu_backprop_last_layer_snp<<<( (dimension1*dimension2)+1023)/1024 , 1024 >>>( dimension1, dimension2, d_para_dev_snp_cellenv, d_cellenv_hidden_var, d_snp);



	// TEST: two-dimension GPU structure (this seems to be better, from 17.355 msecs to 5.773 msecs)
	BLOCK_SIZE = 32;
    // setup execution parameters
	threads = dim3(BLOCK_SIZE, BLOCK_SIZE);
    //grid = dim3( ( WC + threads.x - 1 )/threads.x, ( HC + threads.y - 1 )/threads.y );
    // NOTE: I will make the testing dummy
	grid = dim3( ( dimension2 + threads.x - 1 )/threads.x, ( dimension1 + threads.y - 1 )/threads.y );
	// naive implementation (without deep optimization)
	gpu_backprop_last_layer_snp_2D<<< grid, threads >>>( dimension1, dimension2, d_para_dev_snp_cellenv, d_cellenv_hidden_var, d_snp);



	/*
	// the following routine increased the running time
	// TEST: shared memory scheme
	gpu_backprop_last_layer_snp_2D_sharedmem<<< grid, threads >>>( dimension1, dimension2, grid.x, grid.y, d_para_dev_snp_cellenv, d_cellenv_hidden_var, d_snp);
	*/





	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("cellenv (last layer): Time used totally is %.3f msec.\n", msecTotal);














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
    // Record the start event
	cudaEventRecord(start, NULL);

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
	gpu_backprop_error_prop<<<( num_batch_hidden + 255 )/256 , 256 >>>( dimension1, dimension2, d_para_batch_hidden_gene, d_error_list, d_batch_hidden_var_backerror);
	//
	//gpu_neuralnet_ac_func_dev<<<( num_batch_hidden + 255 )/256 , 256 >>>( num_batch_hidden, d_batch_hidden_var);
	// NOTE: this is not just logistic dev; it's combined with the propogated errors
	gpu_neuralnet_ac_func_dev_error<<<( num_batch_hidden + 255 )/256 , 256 >>>( num_batch_hidden, d_batch_hidden_var, d_batch_hidden_var_backerror);
	//
	dimension1 = matrix_para_batch_batch_hidden.get_dimension1();
	dimension2 = matrix_para_batch_batch_hidden.get_dimension2();
	gpu_backprop_last_layer<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_para_dev_batch_batch_hidden, d_batch_hidden_var, d_batch_var);


	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("batch: Time used totally is %.3f msec.\n", msecTotal);






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





	//============== timing starts ================
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);
	// Record the start event
	cudaEventRecord(start, NULL);

	//=======================================
	//============ GPU Computing ============
	//=======================================
	gpu_penalty_cis<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis , d_list_para_cis_gene[etissue_index], d_list_para_dev_cis_gene[etissue_index], lambda_lasso, lambda_ridge, sigma);

	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	long int dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	//gpu_penalty<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_snp_cellenv, d_para_dev_snp_cellenv, lambda_snp_cellenv, sigma);
	//==============================================
	// NOTE: maximum number of blocks in a grid !!!
	//==============================================
	gpu_penalty<<<( (dimension1*dimension2)+1023)/1024 , 1024 >>>( (dimension1*dimension2) , d_para_snp_cellenv, d_para_dev_snp_cellenv, lambda_snp_cellenv, sigma);

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

	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
    float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("regularization: Time used totally is %.3f msec.\n", msecTotal);





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



	//============== timing starts ================
	// Allocate CUDA events that we'll use for timing
	cudaEvent_t start;
	cudaEventCreate(&start);
	cudaEvent_t stop;
	cudaEventCreate(&stop);
    // Record the start event
	cudaEventRecord(start, NULL);

	//===============================================
	//================ GPU computing ================
	//===============================================
	gpu_gd<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis , d_list_para_cis_gene[etissue_index], d_list_para_dev_cis_gene[etissue_index], rate_learner);

	//matrix_para_dev_snp_cellenv --> float * d_para_dev_snp_cellenv
	long int dimension1 = matrix_para_dev_snp_cellenv.get_dimension1();
	long int dimension2 = matrix_para_dev_snp_cellenv.get_dimension2();
	//gpu_gd<<<( (dimension1*dimension2)+255)/256 , 256 >>>( (dimension1*dimension2) , d_para_snp_cellenv, d_para_dev_snp_cellenv, rate_learner);
	//==============================================
	// NOTE: maximum number of blocks in a grid !!!
	//==============================================
	gpu_gd<<<( (dimension1*dimension2)+1023)/1024 , 1024 >>>( (dimension1*dimension2) , d_para_snp_cellenv, d_para_dev_snp_cellenv, rate_learner);

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

	//============== timing ends ================
    // Record the stop event
	cudaEventRecord(stop, NULL);
    // Wait for the stop event to complete
	cudaEventSynchronize(stop);
	// Timing
    float msecTotal = 0.0f;
	cudaEventElapsedTime(&msecTotal, start, stop);
    // Compute and print the performance
	printf("gd: Time used totally is %.3f msec.\n", msecTotal);




	cout << "[@@] leaving the gradient descent..." << endl;
}








//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//#############################################
//=============================================
// below are loglike and testerror relevant
// not counted toward the speed testing
//=============================================
//#############################################
//$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
//@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@










// calculate the loglikelihood for all the samples in one tissue
float cal_loglike(string etissue)
{
	cout << "[@@] now calculating the log-likelihood..." << endl;

	float loglike = 0;
	loglike = forward_loglike_testerror(0, etissue, &snp_dosage_list, gene_rpkm_exp, cellenv_hidden_var, batch_var, batch_hidden_var);	// indicator=0 --> loglike; indicator=1 --> testerror

	return loglike;
}




// forward process, accumulated the errors
float forward_loglike_testerror(int indicator, string etissue, array<float *, NUM_CHR> * dosage_list_pointer, float * expr_con_pointer, float * cellenv_con_pointer, float * batch_list_pointer, float * batch_hidden_con_pointer)
{
	float result = 0;
	float * d_result;
	checkCudaErrors(cudaMalloc(&d_result, 1*sizeof(float)));
	gpu_clean<<<( 1 + 255 )/256 , 256 >>>( 1 , d_result );

	int etissue_index = etissue_index_map[etissue];

	//=================================================================================================================
	//******************************************* loglike or testerror ************************************************
	//=================================================================================================================
	int amount_sample;
	if(indicator == 0)	// training set, loglike
	{
		amount_sample = esample_tissue_rep[etissue].size();
	}
	else 				// testing set, testerror
	{
		amount_sample = esample_tissue_rep_test[etissue].size();
	}


	for(int pos = 0; pos < amount_sample; pos++)
	{
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
		//cout << "======== current sample #" << pos+1 << ": " << esample << endl;


		int snp_index = d_snp_index_map[individual];
		d_snp = d_snp_list[snp_index];

		int batch_index = d_batch_index_map[esample];
		d_batch_var = d_batch_list[batch_index];

		//=================================================================================================================
		//******************************************* loglike or testerror ************************************************
		//=================================================================================================================
		if(indicator == 0)	// training set, loglike
		{
			int sample_index = d_sample_index_map[esample];
			d_expr = d_sample_list[sample_index];
		}
		else 				// testing set, testerror
		{
			int sample_index = d_sample_test_index_map[esample];
			d_expr = d_sample_test_list[sample_index];
		}





		//========================================================================
		// forward-propogation (cis-; cell env; batch)
		//========================================================================
		//========================================================================
		long int dimension1, dimension2;

		//========================================================================
		// ****************************** [part1] cis- *********************************
		//===============================================
		//================ GPU computing ================
		//===============================================
		//== multiply the input to the matrix
		gpu_matrix_mul_cis_mul<<<( num_para_cis+255)/256 , 256 >>>( num_para_cis, num_gene, d_snp, d_list_para_cis_gene[etissue_index], d_temp_cis_gene,\
								d_cis_para_start, d_cis_para_amount, d_cis_snp_start, d_cis_para_index1);
		//== sum matrix
		gpu_matrix_mul_cis_add<<<(num_gene + 255) / 256 , 256 >>>( num_gene, d_temp_cis_gene, d_gene_rpkm_exp_cis,\
								d_cis_para_start, d_cis_para_amount, d_cis_snp_start);


		// ********************* [part2] cell env relevant parameters *********************
		//===============================================
		//================ GPU computing ================ (Apr.27, has been debugged, for both MM and neural hidden layer)
		//===============================================
		//==== from SNP to cellenv
		dimension1 = matrix_para_snp_cellenv.get_dimension1();
		dimension2 = matrix_para_snp_cellenv.get_dimension2();
		//== multiply the input to the matrix
		//============================
		// NOTE: grid size
		//============================
		//gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_snp, d_para_snp_cellenv, d_temp_snp_cellenv);
		gpu_matrix_mul_mul<<<( (dimension1*dimension2)+1023)/1024 , 1024 >>>( dimension1, dimension2, d_snp, d_para_snp_cellenv, d_temp_snp_cellenv);
		//== sum matrix
		gpu_matrix_mul_add<<<(dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_snp_cellenv, d_cellenv_hidden_var);

		//==== neuralnet
		gpu_neuralnet_ac_func<<<(num_cellenv + 255) / 256 , 256 >>>( num_cellenv , d_cellenv_hidden_var);

		//==== from cellenv to gene expression
		dimension1 = cube_para_cellenv_gene[etissue_index].get_dimension1();
		dimension2 = cube_para_cellenv_gene[etissue_index].get_dimension2();
		//== multiply the input to the matrix
		gpu_matrix_mul_mul<<<( (dimension1*dimension2)+255)/256 , 256 >>>( dimension1, dimension2, d_cellenv_hidden_var, d_list_para_cellenv_gene[etissue_index], d_temp_cellenv_gene);
		//== sum matrix
		gpu_matrix_mul_add<<< (dimension1 + 255) / 256 , 256 >>>( dimension1, dimension2, d_temp_cellenv_gene, d_gene_rpkm_exp_cellenv);


		// ********************* [part3] linear or non-linear batches *********************
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


		// ********************* [end] merge the signal from three pathways here, to expr_con_pointer *********************
		//==========================================================================
		//================ GPU computing (without data transfering) ================
		//==========================================================================
		// 1. merge the three in GPU;
		// 2. calculate the error in GPU
		gpu_merge_list_3<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_gene_rpkm_exp_cis, d_gene_rpkm_exp_cellenv, d_gene_rpkm_exp_batch);
		gpu_error_cal<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_error_list, d_gene_rpkm_exp_cis, d_expr);







		// error is the thing actually needed, and by now we have it
		//=================================================================================================================
		//******************************************* loglike or testerror ************************************************
		//=================================================================================================================
		if(indicator == 0)		// training set, loglike
		{
			// I will use "d_gene_rpkm_exp_cis" as the temp list
			gpu_cal_loglike_func<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_error_list, d_gene_rpkm_exp_cis);
			gpu_cal_loglike_add<<< (1 + 255) / 256 , 256 >>>( 1, num_gene, d_gene_rpkm_exp_cis, d_result);
		}
		else 					// testing error, testerror
		{
			// I will use "d_gene_rpkm_exp_cis" as the temp list
			gpu_cal_testerror_func<<< (num_gene + 255) / 256 , 256 >>>( num_gene, d_error_list, d_gene_rpkm_exp_cis);
			// the same one as loglike func
			gpu_cal_loglike_add<<< (1 + 255) / 256 , 256 >>>( 1, num_gene, d_gene_rpkm_exp_cis, d_result);
		}

	}// leave this sample, move on to next sample (in this tissue)



	// transfer back the final result from GPU (loglike or test error)
	checkCudaErrors(cudaMemcpy(&result, d_result, 1*sizeof(float), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(d_result));


	return result;
}






// testing error for one tissue
// type: MAE or AE (mean absolute error or absolute error)
float cal_testerror(string etissue)
{
	cout << "[@@] now calculating the testing error (for the current tissue)..." << endl;

	float testerror = 0;
	testerror = forward_loglike_testerror(1, etissue, &snp_dosage_list, gene_rpkm_exp, cellenv_hidden_var, batch_var, batch_hidden_var);	// indicator=0 --> loglike; indicator=1 --> testerror

	return testerror;
}


