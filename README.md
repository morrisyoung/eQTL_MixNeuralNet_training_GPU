# eQTL_MixNeuralNet_training_GPU

This is the benchmarking version of CUDA GPU code for the training program, on simulated dataset (10% scale of the real dataset).

Now the speed is 0.00051 secs per sample, if pre-loading the data; if loading data on the fly, it always takes 0.15+ secs, as the cost of opening the PCI bus.
