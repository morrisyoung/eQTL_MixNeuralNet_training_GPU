// includes CUDA runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <math.h>



__device__
void add(float *result, float a, float b, float c)
{
    (*result) = a * b + c;
}


// testing calling the device routines
__global__
void saxpy(int n, float a, float *x, float *y)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
    {
        // normal saxpy
        //y[i] = a*x[i] + y[i];
        add(&(y[i]), a, x[i], y[i]);
    }

}



// scaling with a factor specified
__global__
void gpu_scale(long int dimension, float factor, float * x)
{
    long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension)
    {
		x[i] = factor * x[i];
    }

}



// gradient descent:
//		beta = beta - rate * dev
__global__
void gpu_gd(long int dimension, float * para, float * para_dev, float rate)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension)
    {
		para[i] = para[i] - rate * para_dev[i];
    }

}


// regularization routine
__global__
void gpu_penalty(long int dimension, float * para, float * para_dev, float lambda, float sigma)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < dimension)
    {
    	float beta = para[i];
    	float derivative = beta / sqrt (beta * beta + sigma);  // this is an approximation of the LASSO regularization
    	para_dev[i] += lambda * derivative;
	}

}


// elastic net penalty for cis- regulation
__global__
void gpu_penalty_cis(long int dimension, float * para, float * para_dev, float lambda_lasso, float lambda_ridge, float sigma)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < dimension)
    {
    	float beta = para[i];

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
		para_dev[i] += value;
    }

}


// set all the memory to 0
__global__
void gpu_clean(long int dimension, float * array)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension)
    {
		array[i] = 0;
    }

}



// add 1 to all the elements of a matrix
__global__
void gpu_addone(long int dimension, float * array)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension)
    {
		array[i] += 1;
    }

}




// set a number to all the variables
__global__
void gpu_setnum(long int dimension, float * array, float number)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension)
    {
		array[i] = number;
    }

}




// go through the neuralnet (one parameter tunable)
__global__
void gpu_neuralnet_ac_func(long int dimension, float * array)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension)
    {
		float tune = 0.01;  // original choice is 0.01
		array[i] = 1 / ( 1 + exp( - tune * array[i] ));
    }

}


// this is the derivative of the above activation function
__global__
void gpu_neuralnet_ac_func_dev(long int dimension, float * array)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension)
    {
		float tune = 0.01;  // original choice is 0.01
		array[i] = tune * array[i] * (1 - array[i]);
    }

}




// two-step Matrix Multiplication
__global__
void gpu_matrix_mul_mul(long int dimension1, long int dimension2, float * input, float * para, float * temp)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension1*dimension2)
    {
    	//long int index1 = i / dimension2;
    	long int index2 = i % dimension2;
    	if(index2 == dimension2 - 1)
    	{
    		temp[i] = 1 * para[i];
    	}
    	else
    	{
			temp[i] = input[index2] * para[i];
    	}
    }

}


__global__
void gpu_matrix_mul_add(long int dimension1, long int dimension2, float * temp, float * output)
{
	long int index1 = blockIdx.x*blockDim.x + threadIdx.x;
	if(index1 < dimension1)
	{
		long int pos_start = dimension2 * index1;
		output[index1] = 0;
		for(long int i=0; i<dimension2; i++)
		{
			output[index1] += temp[pos_start + i];
		}

	}

}




// two-step Matrix Multiplication (for cis- part)
__global__
void gpu_matrix_mul_cis_mul(long int dimension, long int dimension1, float * input, float * para, float * temp,\
	long int * d_cis_para_start, long int * d_cis_para_amount, long int * d_cis_snp_start, long int * d_cis_para_index1)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension)
    {
    	// find gene pos
    	/*
		int gene_index = 0;
		if(i < d_cis_para_start[1] || i >= d_cis_para_start[dimension1-1])		// boundary
		{

			if(i < d_cis_para_start[1])
			{
				gene_index = 0;
			}
			else
			{
				gene_index = dimension1 - 1;
			}
		}
		else 		// binary search
		{
			int temp_gene_index = int(dimension1 / 2);
			int amount = int(dimension1 / 4);
			while(1)
			{
				if(i >= d_cis_para_start[temp_gene_index] && i < d_cis_para_start[temp_gene_index + 1])
				{
					break;
				}
				if(i < d_cis_para_start[temp_gene_index])
				{
					temp_gene_index -= amount;
					amount = int(amount / 2) + 1;
					continue;
				}
				if(i >= d_cis_para_start[temp_gene_index + 1])
				{
					temp_gene_index += amount;
					amount = int(amount / 2) + 1;
					continue;
				}
			}
			gene_index = temp_gene_index;

		}
		*/
		/*
		// naive search
		int gene_index = 0;
		while(1)
		{
			if(gene_index == dimension1-1)
				break;
			if(i >= d_cis_para_start[gene_index] && i < d_cis_para_start[gene_index+1])		// boundary
				break;

			gene_index += 1;
		}
		*/

		int gene_index = d_cis_para_index1[i];

    	// find shift
		long int shift = i - d_cis_para_start[gene_index];

    	// find snp pos
		long int pos_snp = d_cis_snp_start[gene_index] + shift;

    	// calcluate
		float snp;
		long int amount = d_cis_para_amount[gene_index];
		if(shift == amount - 1)		// intercept
		{
			snp = 1;
		}
		else
		{
			snp = input[pos_snp];
		}

		temp[i] += para[i] * snp;


    }

}

__global__
void gpu_matrix_mul_cis_add(long int dimension1, float * temp, float * output,\
	long int * d_cis_para_start, long int * d_cis_para_amount, long int * d_cis_snp_start)
{
	long int index1 = blockIdx.x*blockDim.x + threadIdx.x;
	if(index1 < dimension1)
	{
		long int pos_start = d_cis_para_start[index1];
		long int dimension2 = d_cis_para_amount[index1];
		output[index1] = 0;
		for(long int i=0; i<dimension2; i++)
		{
			output[index1] += temp[pos_start + i];
		}
	}

}




// merge two arrays to the first one in place
__global__
void gpu_merge_list(long int dimension, float * array1, float * array2)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < dimension)
	{
		array1[i] += array2[i];
	}

}


// merge two arrays to the first one in place
__global__
void gpu_merge_list_3(long int dimension, float * array1, float * array2, float * array3)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < dimension)
	{
		array1[i] += array2[i];
		array1[i] += array3[i];
	}

}



__global__
void gpu_error_cal(long int dimension, float * d_error_list, float * expr_exp, float * expr_real)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < dimension)
	{
		d_error_list[i] = expr_exp[i] - expr_real[i];
	}

}



// backpropogation last layer
// pseudo: (expected rpkm - real rpkm) * cell_env
// pseudo: (expected rpkm - real rpkm) * hidden batch
__global__
void gpu_backprop_last_layer(long int dimension1, long int dimension2, float * para_dev, float * error_list, float * hidden_var)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < dimension1 * dimension2)
	{
    	long int index1 = i / dimension2;
    	long int index2 = i % dimension2;
    	float error = error_list[index1];
    	float cellenv;
    	if(index2 == dimension2-1)
    	{
    		cellenv = 1;					// we do have the intercept term
    	}
    	else
    	{
    		cellenv = hidden_var[index2];
    	}
    	para_dev[i] += error * cellenv;
	}

}


// calculate the errors propogated from last layer
__global__
void gpu_backprop_error_prop(long int dimension1, long int dimension2, float * para, float * error_list, float * output)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < dimension2 - 1)				// we only consider actual hidden factors other than intercept
	{
		long int pos_start = i;
		output[i] = 0;
    	for(long int j=0; j<dimension1; j++)
    	{
    		output[i] += error_list[j] * para[pos_start + j * dimension2];
    	}
	}

}



// TODO: to further check the correctness of this function
// backpropogation for cis- part
__global__
void gpu_backprop_cis(long int dimension, long int dimension1, float * para_dev, float * error_list, float * input,\
	long int * d_cis_para_start, long int * d_cis_para_amount, long int * d_cis_snp_start, long int * d_cis_para_index1)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < dimension)
	{
    	// find gene pos
    	/*
		int gene_index = 0;
		if(i < d_cis_para_start[1] || i >= d_cis_para_start[dimension1-1])		// boundary
		{

			if(i < d_cis_para_start[1])
			{
				gene_index = 0;
			}
			else
			{
				gene_index = dimension1 - 1;
			}
		}
		else 		// binary search
		{
			int temp_gene_index = int(dimension1 / 2);
			int amount = int(dimension1 / 4);
			while(1)
			{
				if(i >= d_cis_para_start[temp_gene_index] && i < d_cis_para_start[temp_gene_index + 1])
				{
					break;
				}
				if(i < d_cis_para_start[temp_gene_index])
				{
					temp_gene_index -= amount;
					amount = int(amount / 2) + 1;
					continue;
				}
				if(i >= d_cis_para_start[temp_gene_index + 1])
				{
					temp_gene_index += amount;
					amount = int(amount / 2) + 1;
					continue;
				}
			}
			gene_index = temp_gene_index;

		}
		*/
		/*
		// naive search
		int gene_index = 0;
		while(1)
		{
			if(gene_index == dimension1-1)
				break;
			if(i >= d_cis_para_start[gene_index] && i < d_cis_para_start[gene_index+1])		// boundary
				break;

			gene_index += 1;
		}
		*/

		int gene_index = d_cis_para_index1[i];

    	// find shift
		long int shift = i - d_cis_para_start[gene_index];

    	// find snp pos
		long int pos_snp = d_cis_snp_start[gene_index] + shift;

    	// calcluate
		float snp;
		long int amount = d_cis_para_amount[gene_index];
		if(shift == amount - 1)		// intercept
		{
			snp = 1;
		}
		else
		{
			snp = input[pos_snp];
		}
		float error = error_list[gene_index];

		para_dev[i] += snp * error;
	}

}


