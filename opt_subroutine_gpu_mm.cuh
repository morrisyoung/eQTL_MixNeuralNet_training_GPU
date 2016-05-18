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


