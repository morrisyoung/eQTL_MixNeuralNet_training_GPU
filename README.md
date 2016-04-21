# eQTL_MixNeuralNet_training_GPU

This is the benchmarking version of CUDA GPU code for the training program, on simulated dataset.

I don't have gene expression data and batch data actually loaded into the GPU, but I can do that, and that makes no difference in benchmarking.

Now the speed is 0.00051 secs per sample, if pre-loading the data; if loading data on the fly, it always takes 0.15+ secs, as the cost of opening the PCI bus.
