# Makefile for CSC461 Program 3

# Author: Andrew Stelter, Jared Johnson
# Class:  CSC461 Programming Languages
# Date:   Fall 2016

# Usage:  make target1 target2 ...

CC=g++
NVCC=nvcc

#-----------------------------------------------------------------------
CFLAGS = -Wall -std=c++11 -fopenmp
CXXFLAGS = $(CFLAGS)
CUDAFLAGS = -std=c++11

LIBS = -lm -fopenmp
#-----------------------------------------------------------------------
# Specific targets:

# MAKE allows the use of "wildcards", to make writing compilation instructions
# a bit easier. GNU make uses $@ for the target and $^ for the dependencies.

all:    cuda

cudaPrime.o: cudaPrime.cu
	$(NVCC) $(CUDAFLAGS) -c $^
    
cuda: CXXFLAGS += -D_CUDA_PRIME 
cuda: main.o sequentialPrime.o ompPrime.o asyncPrime.o cudaPrime.o functions.o
	$(NVCC) -o primes $^ -Xcompiler "$(LIBS)"

nocuda:	main.o sequentialPrime.o ompPrime.o asyncPrime.o functions.o
	$(CC) -o primes $^

clean:
	rm -f *.o *~ *.wrd *.csv *.data .nfs* primes

