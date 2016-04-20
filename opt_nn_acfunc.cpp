// the activation function of the neural network


#include <math.h>       /* exp */
#include "opt_nn_acfunc.h"




// let's define the tune parameter here
//$$$$$$$$$$$ we need the tune parameter to tune the signal passed from upper layer $$$$$$$$$$$$
float tune = 0.01;  // original choice is 0.01




// here we have an activation function for the neural network
// function: takes the arrayed input value and wrench them with a *activation* function (like logistic function)
void neuralnet_ac_func(float * array, int length)
{
	for(int i=0; i<length; i++)
	{
		array[i] = 1 / ( 1 + exp( - tune * array[i] ));
	}
}



// this is the derivative of the above activation function
float neuralnet_ac_func_dev(float input)
{
	return tune * input * (1 - input);
}

