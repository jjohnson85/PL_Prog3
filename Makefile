# Makefile for CSC461 Program 3

# Author: Andrew Stelter, Jared Thompson
# Class:  CSC461 Programming Languages
# Date:   Fall 2016

# Usage:  make target1 target2 ...

#-----------------------------------------------------------------------
CFLAGS = -Wall -fopenmp -lm -std=c++11

#-----------------------------------------------------------------------
# Specific targets:

# MAKE allows the use of "wildcards", to make writing compilation instructions
# a bit easier. GNU make uses $@ for the target and $^ for the dependencies.

all:    cuda

cuda:   sequentialPrime.o ompPrime.o asyncPrime.o cudaPrime.o functions.o main.o
	nvcc -o primes $^ $(CFLAGS) -DCUDAPRIME

nocuda:	sequentialPrime.o ompPrime.o asyncPrime.o functions.o main.o
	g++ -o primes $^ $(CFLAGS)

clean:
	rm -f *.o *~ *.wrd *.csv *.data .nfs* primes

