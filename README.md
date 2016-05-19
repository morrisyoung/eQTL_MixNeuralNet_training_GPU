# eQTL_MixNeuralNet_training_GPU

This is the benchmarking version of CUDA GPU code for the training program, on simulated dataset (10% scale of the real dataset).

Now the speed is 50.258 msec per sample (pre-loading the data into GPU memory). This will be the baseline for the further optimization (as the implementation is very naive for this version).

(May.19, 16) I used shared memory optimization, now the speed per sample is 30.285 msec.

