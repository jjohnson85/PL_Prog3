#include<cstdio>
#include<iostream>
#include<cmath>
#include<string>
#include "cudaPrime.cuh"

#define WARPSIZE 32
#define MAXTHREADS 1024
#define MAXBLOCKS 65535

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
__global__ void isPrimeFine( ull val, ull offset, int * result )
{
    //+2 to ignore 0, and 1
    bool prime = true;
    ull tid = threadIdx.x + blockIdx.x * blockDim.x + offset;
    if(tid == 0) *result = 1;
    if(val < 1) prime = false;
    else if(tid > 1 && tid < val)
    {
        if(val % tid == 0)
        {
            prime = false;
        }
    }
    if(tid == 0)
        if(__any(!prime)) *result = 0;
}

//test for a numbers primality
__global__ void isPrimeHybrid( ull store_offset, ull val_offset, int * result )
{
    ull tid = threadIdx.x;
    ull val = blockIdx.x + val_offset;
    ull idx = blockIdx.x + store_offset;

    if(tid == 0)result[idx] = 1;
    for(ull i=tid; i<val; i+=blockDim.x)
    {
        if(val % i == 0) result[idx] = 0;
    }
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
                //if(data[idx+offset] > 10000)
                //printf("Thread %lli adding index %lli: %i\n", tid, idx+offset, data[idx+offset]);
                data[idx] += data[idx+offset];
            }
            offset >>= 1;
        }
    }
}

string getCudaDeviceProperties()
{
    char out[500];
    int deviceNum;
    cudaGetDevice( &deviceNum );
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, deviceNum);
    
    sprintf(out, "%s, CUDA %i.%i, %lli Mbytes global memory, %i CUDA cores", properties.name, properties.major, properties.minor, properties.totalGlobalMem/1024/1024, properties.maxThreadsPerBlock);
    return out;
}

ull sumRange(int * data, ull size, int warps)
{   
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
	cudaError_t lastError;
	
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
            blocksDone += blocks;
        }
        threadRange = WARPSIZE*threadRange;
        threadsNeeded = (ull)(ceil((double)size/threadRange)+0.2);
        warpsNeeded = (ull)(ceil((double)threadsNeeded/WARPSIZE)+0.2);
	    blocksNeeded = (ull)(ceil((double)warpsNeeded/warps)+0.2);
        blocksDone = 0;
        
        cudaDeviceSynchronize();
        lastError = cudaPeekAtLastError();
        if(lastError != cudaSuccess)
        {
            cout << "sumRange::Error during kernel execution: " << cudaGetErrorString(lastError) << endl;
            return 0;
        }
    }while(more);

    cudaMemcpy(&result, data, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}

int runCudaCoarse( ull start, ull end, unsigned int warps )
{
	ull range = end-start+1;
	ull size = (range)*sizeof(int);
	ull threadsPer = warps*WARPSIZE;
	if(threadsPer > MAXTHREADS) threadsPer = MAXTHREADS;
	
	ull totalBlocks = (ull)(ceil((double)range/threadsPer)+0.2);
	cudaError_t lastError;
	
	int *d_result;
	if(cudaMalloc( &d_result, size )==cudaErrorMemoryAllocation)
	{
	    cout << "Error allocating memory on cuda device" << endl;
	    return 0;
	}
	int *result = (int *)malloc( size );
	
    ull threadsThisTime;
    ull totalDone=0;
    ull blocks;
	while(totalDone < range)
	{
	    threadsThisTime = MAXBLOCKS*threadsPer;
	    if(start+threadsThisTime > end) threadsThisTime = end-start+1;
	    blocks = (ull)(ceil(threadsThisTime/(double)threadsPer)+0.2);

        //cout << "IsPrimeCoarse<<<" << blocks << ", " << threadsPer << ">>>(" << totalDone << ", " << start << ", " << end << ", " << d_result << ")" << endl;
		isPrimeCoarse<<< blocks , threadsPer >>>( totalDone, start, end, d_result ); 
		    
		start += threadsThisTime;
		totalDone += threadsThisTime;
	}

    cudaDeviceSynchronize();
    lastError = cudaPeekAtLastError();
    if(lastError != cudaSuccess)
    {
        cout << "runCudaCoarse::Error during kernel execution: " << cudaGetErrorString(lastError) << endl;
        cudaFree( d_result );
	    free( result );
        return 0;
    }
    
    int count = sumRange(d_result, range, warps);

	cudaFree( d_result );
	free( result );
	
	return count;
}

int runCudaFine( ull start, ull end, unsigned int warps )
{
    ull range = end-start+1;
    ull size = (range)*sizeof(int);
    ull threadsPer = warps*WARPSIZE;
    
    if(threadsPer > MAXTHREADS) threadsPer = MAXTHREADS;
	cudaError_t lastError;
    	
    int* d_result;
	if(cudaMalloc( &d_result, size )==cudaErrorMemoryAllocation)
	{
	    cout << "Error allocating memory on cuda device" << endl;
	    return 0;
	}
    int* result = (int *)malloc( size );
        
    ull blocks;
    //i - index of return location
    //j - value to test
    int* i = d_result;
    for(ull j=start; j<=end; i++, j++)
    {
        //k - Offset from 0 for thread indices
        for(ull k=0; k<j; k+= threadsPer*MAXBLOCKS)
        {
            blocks = (ull)(ceil((double)j/threadsPer)+0.2);
            if(blocks > MAXBLOCKS) blocks = MAXBLOCKS;
            
            //cout << "IsPrimeFine<<<" << blocks << ", " << threadsPer << ">>>(" << j << ", " << k << ", " << i << ")" << endl;
            isPrimeFine<<<blocks, threadsPer>>>( j, k, i );
        }
    }

    cudaDeviceSynchronize();
    lastError = cudaPeekAtLastError();
    if(lastError != cudaSuccess)
    {
        cout << "runCudaFine::Error during kernel execution: " << cudaGetErrorString(lastError) << endl;
        cudaFree( d_result );
	    free( result );
        return 0;
    }
    
    int count = sumRange(d_result, range, warps);
    
    cudaFree( d_result );
    free( result );
    
    return count;
}

int runCudaHybrid( ull start, ull end, unsigned int warps )
{
    ull range = end-start+1;
    ull size = (range)*sizeof(int);
    ull threadsPer = warps*WARPSIZE;
    
    if(threadsPer > MAXTHREADS) threadsPer = MAXTHREADS;
	cudaError_t lastError;
	
    int* d_result;
	if(cudaMalloc( &d_result, size )==cudaErrorMemoryAllocation)
	{
	    cout << "Error allocating memory on cuda device" << endl;
	    return 0;
	}
    int* result = (int *)malloc( size );
    
    ull blocks;
    for(ull i=start, j=0; i<end; i+= MAXBLOCKS, j+=MAXBLOCKS)
    {
        blocks = end-i+1;
        if(blocks > MAXBLOCKS) blocks = MAXBLOCKS;
        
        //cout << "IsPrimeHybrid<<<" << blocks << ", " << threadsPer << ">>>(" << j << ", " << i << ", " << d_result << ")" << endl;
        isPrimeHybrid<<<blocks, threadsPer>>>(j, i, d_result);
    }

    cudaDeviceSynchronize();
    lastError = cudaPeekAtLastError();
    if(lastError != cudaSuccess)
    {
        cout << "runCudaHybrid::Error during kernel execution: " << cudaGetErrorString(lastError) << endl;
        cudaFree( d_result );
	    free( result );
        return 0;
    }
    int count = sumRange(d_result, range, warps);
    
    cudaFree( d_result );
    free( result );
    
    return count;
}
