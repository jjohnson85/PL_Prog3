# Makefile for CSC461 Program 3

# Author: Andrew Stelter, Jared Johnson
# Class:  CSC461 Programming Languages
# Date:   Fall 2016

# Usage:  make target1 target2 ...

CC=g++
NVCC=nvcc

#-----------------------------------------------------------------------
CFLAGS = -Wall -fopenmp -lm -std=c++11
CXXFLAGS = $(CFLAGS)
CUDAFLAGS = -std=c++11

LIBS = -lcudart
LIBDIRS = -L/usr/local/cuda/lib64
#-----------------------------------------------------------------------
# Specific targets:

# MAKE allows the use of "wildcards", to make writing compilation instructions
# a bit easier. GNU make uses $@ for the target and $^ for the dependencies.

all:    cuda

cudaPrime.o: cudaPrime.cu
	$(NVCC) $(CUDAFLAGS) -c cudaPrime.cu
    
cuda: CXXFLAGS += -D_CUDA_PRIME 
cuda: sequentialPrime.o ompPrime.o asyncPrime.o cudaPrime.o functions.o main.o 
	$(CC) -o primes $^ $(CXXFLAGS) $(LIBDIRS) $(LIBS)

nocuda:	sequentialPrime.o ompPrime.o asyncPrime.o functions.o main.o
	$(CC) -o primes $^ $(CXXFLAGS)

clean:
	rm -f *.o *~ *.wrd *.csv *.data .nfs* primes

