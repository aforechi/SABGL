#!/bin/bash
# Download MNIST dataset
if [ ! -f "train-images-idx3-ubyte" ]; then

	wget https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz 
	wget https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
	wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
	wget https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz

	gunzip train-images-idx3-ubyte.gz
	gunzip train-labels-idx1-ubyte.gz
	gunzip t10k-images-idx3-ubyte.gz
	gunzip t10k-labels-idx1-ubyte.gz
else
    echo "MNIST dataset already downloaded."
fi
