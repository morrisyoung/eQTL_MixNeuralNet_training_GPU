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

