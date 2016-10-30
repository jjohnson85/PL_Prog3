
#include<iostream>
#include<cmath>

#include "cudaPrime.cuh"

using namespace std;

//test for a numbers primality
__global__ void isPrime( unsigned long long p, unsigned long long end, int * result )
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int test = tid + p;

    if(test < end)
    {
	    bool prime = true;

	    if( test < 2 )
	    {
		    prime = false;
		    result[tid] = false;
		    return;
	    }

	    for( int i = 2; i < test / 2; i++ )
	    {
		    if( test % i == 0 )
		    {
			    prime = false;
		    }
	    }

	    result[tid] = prime;
	}
}

int runCuda( unsigned long long start, unsigned long long end )
{
	unsigned int range = end-start;
	unsigned int lastRangeTested = 0;
	unsigned int size = (range)*sizeof(int);
	unsigned int numblocks = 60000;
	unsigned int numcalls = 1;
	unsigned int numthreads = 32;
	int count = 0;

	int *result = (int *)malloc( size );
	int *d_result;
	cudaMalloc( (int**)&d_result, size );
	//65535 max blocks

	numblocks = range;
	while( numblocks > 60000 )
	{
		numblocks -= 60000;
		numcalls += 1;
	}

	numblocks = range;
	for( int i = 0; i < numcalls; i++ )
	{
		isPrime<<< (int)ceil(numblocks/(double)numthreads) , numthreads >>>( i, end, d_result ); 
		lastRangeTested += numblocks * 32;
		numblocks -= 60000;
	}

	cudaMemcpy( result, d_result, size, cudaMemcpyDeviceToHost );

	for( int i = 0; i < range; i++ )
	{
		if( result[i] == true )
		{
			count++;
		}
	}

	cudaFree( d_result );
	free( result );
	return count;
}
