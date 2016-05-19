// there are various kinds of Matrix Multiplication methods to be called.

/*
 * 5KK73
 * Eindhoven University of Technology
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

// includes CUDA runtime
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h> // helper functions for SDK examples
#include <math.h>



////////////////////////////////////////////////////////////////////////////////
//! Matrix multiplication on the device: C = A * B
//! wA is A's width and wB is B's width
////////////////////////////////////////////////////////////////////////////////
__global__ void
matrixMul_naive( float * C, float * A, float * B, long int hA, long int wA, long int wB)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// Accumulate row i of A and column j of B
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	// DEBUG
	//printf("%ld, %ld\n", i, j);

	if(i < hA && j < wB)
	{

		// DEBUG
		//printf("%ld, %ld\n", i, j);


		float accu = 0.0;

		for(long int k=0; k<wA; k++)		// we have the intercept (1) in the snp list
		{
			accu += A[ i * wA + k ] * B[ k * wB + j ];
		}

		// Write the block sub-matrix to device memory;
		// each thread writes one element
		C[ i * wB + j ] = accu;
	}

}





// the following two routines are for optimizing the huge MM with schemes:
//	1. shared memory (multiple data access)
//	2. parallel computation of subsets
//=============================================================================================================
__global__ void
matrixMul_upgrade_multi(long int dimension1, long int dimension2, int L, int num_block,\
						float * d_cellenv_hidden_var_sub, float * d_para_snp_cellenv, float * d_snp)
{

	if(blockIdx.x == num_block - 1)					// the last block
	{
		if(threadIdx.x < dimension1)
		{
			//==== shared memory
			int length = dimension2 - blockIdx.x*L;
			//__shared__ float A[length];
			__shared__ float A[1];					// TODO: to manually set up

			// load dosage data using all the threads
			for(int j=threadIdx.x; j<length; j+=dimension1)
			{
				long int pos = blockIdx.x*L + j;
				A[j] = d_snp[pos];
			}

			// wait to finish loading
			__syncthreads();

			//==== computation
			long int index = blockIdx.x*dimension1 + threadIdx.x;
			float temp = 0;
			long int start = threadIdx.x*dimension2 + blockIdx.x*L;
			for(int j=0; j<length; j++)
			{
				temp += d_para_snp_cellenv[start + j]*A[j];
			}
			d_cellenv_hidden_var_sub[index] = temp;
		}

	}
	else 											// the usual block
	{
		if(threadIdx.x < dimension1)
		{
			//==== shared memory
			int length = L;
			//__shared__ float A[length];
			__shared__ float A[10000];				// TODO: to manually set up

			// load dosage data using all the threads
			for(int j=threadIdx.x; j<length; j+=dimension1)
			{
				long int pos = blockIdx.x*L + j;
				A[j] = d_snp[pos];
			}

			// wait to finish loading
			__syncthreads();

			//==== computation
			long int index = blockIdx.x*dimension1 + threadIdx.x;
			float temp = 0;
			long int start = threadIdx.x*dimension2 + blockIdx.x*L;
			for(int j=0; j<length; j++)
			{
				temp += d_para_snp_cellenv[start + j]*A[j];
			}
			d_cellenv_hidden_var_sub[index] = temp;

		}
	}

}




__global__ void
matrixMul_upgrade_sum(long int dimension1, long int num_block, float * d_cellenv_hidden_var_sub, float * d_cellenv_hidden_var)
{
	long int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < dimension1)
    {
    	float value = 0;
    	for(long int j=i; j<dimension1*num_block; j+=dimension1)
    	{
    		value += d_cellenv_hidden_var_sub[j];
    	}
    	d_cellenv_hidden_var[i] = value;
    }

}
//=============================================================================================================





__global__ void
gpu_backprop_last_layer_snp_2D(long int hA, long int wA, float * para_dev, float * error_list, float * d_snp)
{
	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// Accumulate row i of A and column j of B
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	if(i < hA && j < wA)
	{
    	float error = error_list[i];
    	float dosage = d_snp[j];		// we do have the intercept term
    	para_dev[ i * wA + j ] += error * dosage;
	}

}



// the following is optimized with shared memory scheme
//=============================================================================================================
__global__ void
gpu_backprop_last_layer_snp_2D_sharedmem(long int hA, long int wA, int block_x, int block_y, float * para_dev, float * error_list, float * d_snp)
{

	// NOTE: all the TODO are based on BLOCK SIZE 32*32 !!!


	// Block index
	long int bx = blockIdx.x;
	long int by = blockIdx.y;

	// Thread index
	long int tx = threadIdx.x;
	long int ty = threadIdx.y;

	// Accumulate row i of A and column j of B
	long int i = by * blockDim.y + ty;
	long int j = bx * blockDim.x + tx;

	if(i < hA && j < wA)
	{
		if(bx == block_x-1 && by == block_y-1)
		{
			// shared memory processing
			int length1 = 1;			// TODO
			int length2 = 16;			// TODO
			__shared__ float A[1];		// TODO
			__shared__ float B[16];		// TODO

			long int start_x = bx * blockDim.x;
			long int start_y = by * blockDim.y;
			if(ty == 0)		// load x array
			{
				A[tx] = d_snp[start_x + tx];
			}
			if(tx == 0)
			{
				B[ty] = error_list[start_y + ty];
			}

			__syncthreads();


			// computation
			long int pos = i*wA + j;
			para_dev[pos] += B[ty] * A[tx];

		}

		if(bx == block_x-1 && by != block_y-1)
		{
			// shared memory processing
			int length1 = 1;			// TODO
			int length2 = 32;			// TODO
			__shared__ float A[1];		// TODO
			__shared__ float B[32];		// TODO

			long int start_x = bx * blockDim.x;
			long int start_y = by * blockDim.y;
			if(ty == 0)		// load x array
			{
				A[tx] = d_snp[start_x + tx];
			}
			if(tx == 0)
			{
				B[ty] = error_list[start_y + ty];
			}

			__syncthreads();


			// computation
			long int pos = i*wA + j;
			para_dev[pos] += B[ty] * A[tx];
		}

		if(bx != block_x-1 && by == block_y-1)
		{
			// shared memory processing
			int length1 = 32;			// TODO
			int length2 = 16;			// TODO
			__shared__ float A[32];		// TODO
			__shared__ float B[16];		// TODO

			long int start_x = bx * blockDim.x;
			long int start_y = by * blockDim.y;
			if(ty == 0)		// load x array
			{
				A[tx] = d_snp[start_x + tx];
			}
			if(tx == 0)
			{
				B[ty] = error_list[start_y + ty];
			}

			__syncthreads();


			// computation
			long int pos = i*wA + j;
			para_dev[pos] += B[ty] * A[tx];
		}

		if(bx != block_x-1 && by != block_y-1)
		{
			// shared memory processing
			int length1 = 32;			// TODO
			int length2 = 32;			// TODO
			__shared__ float A[32];		// TODO
			__shared__ float B[32];		// TODO

			long int start_x = bx * blockDim.x;
			long int start_y = by * blockDim.y;
			if(ty == 0)		// load x array
			{
				A[tx] = d_snp[start_x + tx];
			}
			if(tx == 0)
			{
				B[ty] = error_list[start_y + ty];
			}

			__syncthreads();


			// computation
			long int pos = i*wA + j;
			para_dev[pos] += B[ty] * A[tx];
		}

	}

}
//=============================================================================================================


