#include<cstdio>
#include<iostream>
#include<cmath>

#include "cudaPrime.cuh"

#define WARPSIZE 32
#define MAXTHREADS 1024
#define MAXBLOCKS 65536

using namespace std;

//test for a numbers primality
__global__ void isPrimeCoarse( ull offset, ull start, ull end, int * result )
{
	ull tid = threadIdx.x + blockIdx.x * blockDim.x;
	ull idx = tid + offset;
	ull val = tid + start;

    int prime = 1;
    if(val <= end)
    {
	    if( val < 2 )
	    {
		    prime = 0;
	    }
        else
        {
	        for( int i = 2; i < val / 2; i++ )
	        {
		        if( val % i == 0 )
		        {
			        prime = 0;
		        }
	        }
	    }
	    
        result[idx] = prime;
	}
}

//test for a numbers primality
__global__ void isPrimeFine( ull offset, ull start, ull end, int * result )
{

}

//test for a numbers primality
__global__ void isPrimeHybrid( ull offset, ull start, ull end, int * result )
{

}

__global__ void reduce( int * data, ull size, ull gapSize )
{
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    ull idx = tid*gapSize;
    ull offset = 16*gapSize;
    
    if(idx < size)
    {
        //printf("Thread %lli writing to index %lli\n", tid, idx);
        while(offset >= gapSize)
        {
            //printf("Offset %lli : %lli -> %lli\n", offset, idx, idx+offset);
            if(idx+offset < size)
            {
                //printf("Thread %lli adding index %lli: %i\n", tid, idx+offset, data[idx+offset]);
                data[idx] += data[idx+offset];
            }
            offset >>= 1;
        }
    }
}

ull sumRange(int * data, ull size, int warps)
{
    cudaDeviceSynchronize();
    
    //cout << "\nStarting cuda range sum on " << size << " items" << endl;

    int result;
	ull threadsPer = WARPSIZE*warps;
	
    bool more=false;
    ull threadRange = 1;
    ull threadsNeeded = (ull)(ceil((double)size/threadRange)+0.2);
    ull warpsNeeded = (ull)(ceil((double)threadsNeeded/WARPSIZE)+0.2);
	ull blocksNeeded = (ull)(ceil((double)warpsNeeded/warps)+0.2);
	ull blocks;
	ull blocksDone = 0;
	
    do
    {
        //We'll need to reduce down to running
        //In a single warp
        more = warpsNeeded > 1;
        //cout << "Iteration " << count++ << endl;
        
        while(blocksDone < blocksNeeded)
        {
            //cout << blocksDone << "/" << blocksNeeded << " blocks" << endl;
            //Check if everything can be summed in one go, or if 
            //Multiple kernel calls needed
            if(blocksNeeded < MAXBLOCKS) blocks = blocksNeeded;
            else
            {
                blocks = MAXBLOCKS;
            }
            
            //cout << "Executing " << blocks << " blocks with " << threadsPer << " each" << endl;
            //cout << "Start is data[" << blocksDone*threadsPer << "], Size is " << size-blocksDone*threadsPer << " range is " << threadRange << endl;
            reduce<<<blocks, threadsPer>>>(data+blocksDone*threadsPer, size-blocksDone*threadsPer, threadRange);
            cudaDeviceSynchronize();
            blocksDone += blocks;
        }
        threadRange = WARPSIZE*threadRange;
        threadsNeeded = (ull)(ceil((double)size/threadRange)+0.2);
        warpsNeeded = (ull)(ceil((double)threadsNeeded/WARPSIZE)+0.2);
	    blocksNeeded = (ull)(ceil((double)warpsNeeded/warps)+0.2);
        blocksDone = 0;
    }while(more);

    cudaMemcpy(&result, data, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}

int runCudaCoarse( ull start, ull end, unsigned int warps )
{
	ull range = end-start+1;
	ull size = (range)*sizeof(int);
	ull threadsPer = warps*WARPSIZE;
	int count = 0;
	if(threadsPer > MAXTHREADS) threadsPer = MAXTHREADS;
	
	ull totalBlocks = (ull)(ceil((double)range/threadsPer)+0.2);

	int *result = (int *)malloc( size );
	int *d_result;
	cudaMalloc( &d_result, size );

    ull threadsThisTime;
    ull totalDone=0;
    int blocks;
	while(totalDone < range)
	{
	    threadsThisTime = MAXBLOCKS*threadsPer;
	    if(start+threadsThisTime > end) threadsThisTime = end-start+1;
	    blocks = (ull)(ceil(threadsThisTime/(double)threadsPer)+0.2);

		isPrimeCoarse<<< blocks , threadsPer >>>( totalDone, start, end, d_result ); 
		    
		start += threadsThisTime;
		totalDone += threadsThisTime;
	}

    count = sumRange(d_result, range, warps);

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
