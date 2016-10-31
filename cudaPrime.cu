
#include<iostream>
#include<cmath>

#include "cudaPrime.cuh"

#define WARPSIZE 32
#define MAXTHREADS 1024
#define MAXBLOCKS 65536

using namespace std;

//test for a numbers primality
__global__ void isPrimeCoarse( ull offset, ull end, int * result )
{
	ull tid = threadIdx.x + blockIdx.x * blockDim.x;
	ull test = tid + offset;

    bool prime = true;
    if(test <= end)
    {
	    if( test < 2 )
	    {
		    prime = false;
		    result[tid] = false;
	    }
        else
        {
	        for( int i = 2; i < test / 2; i++ )
	        {
		        if( test % i == 0 )
		        {
			        prime = false;
		        }
	        }
            
	        result[test] = prime;
	    }
	}
}

int runCudaCoarse( ull start, ull end, unsigned int warps )
{
	ull range = end-start;
	ull size = (range)*sizeof(int);
	ull threadsPer = warps*WARPSIZE;
	int count = 0;
	if(threadsPer > MAXTHREADS) threadsPer = MAXTHREADS;
	
	ull totalBlocks = (int)(ceil((double)range/threadsPer)+0.2);

	int *result = (int *)malloc( size );
	int *d_result;
	cudaMalloc( (int**)&d_result, size );

    ull threadsThisTime, offset=start;
    int blocks;
	while(offset <= end)
	{
	    threadsThisTime = MAXBLOCKS*threadsPer;
	    if(offset+threadsThisTime > end) threadsThisTime = end-offset+1;
	    blocks = (int)(ceil(threadsThisTime/(double)threadsPer)+0.2);

		isPrimeCoarse<<< blocks , threadsPer >>>( offset, end, d_result ); 
		
		offset += blocks*threadsPer;
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

int runCudaFine( ull start, ull end, unsigned int warps )
{

    return 0;
}

int runCudaHybrid( ull start, ull end, unsigned int warps )
{

    return 0;
}
