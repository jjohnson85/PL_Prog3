#include<cstdio>
#include<iostream>
#include<cmath>
#include<string>
#include "cudaPrime.cuh"

#define WARPSIZE 32
#define MAXTHREADS 1024
#define MAXBLOCKS 65535

using namespace std;

/*
 * isPrimeCoarse
 * 
 * Goarse-grained GPGPU kernel for finding
 * prime numbers. Each thread takes a number
 * in the range and checks if it is prime 
 * by dividing by all numbers up to n/2
 *
 * @params
 *      [in] start - The bottom of the range to find primes in
 *      [in] end - The top of the range to find primes in
 *      [out] result - Int array for storing results
 *          Result values which are set to 1 represent the
 *          associated value being non-prime
 */
__global__ void isPrimeCoarse( ull start, ull end, int* result )
{
    //Find thread id
	ull tid = threadIdx.x + blockIdx.x * blockDim.x;
	
    //Find value to check
    ull val = tid + start;

    //Value is by default prime
    char prime = 0;
    if(val <= end)
    {
        //Values < 2 are non prime
	    if( val < 2 )
	    {
		    prime = 1;
	    }
        else
        {
            //Otherwise, check if any number
            //up to n/2 divides n
	        for( ull i = 2; i < val / 2; i++ )
	        {
		        if( val % i == 0 )
		        {
			        prime = 1;
		        }
	        }
	    }
        //Store the result in the array
        result[tid] = prime;
	}
}

/*
 * isPrimeFine
 *
 * Fine-grained GPGPU kernel for finding
 * prime numbers. Each thread checks if one
 * specific number divides the input. 
 *
 * @params
 *      [in] val - The value to check for primality
 *      [in] offset - The amount to add to the threadid
            to find the number to divide val by
 *      [out] result - Int pointer associated with val
 *          NOTE - The kernel assumes this value is 0,
 *              and it is set to 1 if any thread finds
 *              a number which divides val
 */
__global__ void isPrimeFine( ull val, ull offset, int * result )
{
    //Assume the value is prime
    bool prime = true;

    //Find the value to divide by
    //The extra +2 is to not divide by 0 or 1
    ull tid = threadIdx.x + blockIdx.x * blockDim.x + offset + 2;

    //Non prime if < 2
    if(val < 2)
    {
        if(tid == 2) prime = false;
    }
    else if(tid < val/2)
    {
        //Check if the value the thread has divides val
        if(val % tid == 0)
        {
            prime = false;
        }
    }
    //Merge all the results of each warp
    //If any found non-prime, set prime to not-prime
    prime = !__any(!prime);

    //Every first thread in its warp sets
    //the result pointer if it's warp found
    //val to be non-prime. This is to cut down
    //on memory accesses
    if(threadIdx.x%32==0 && !prime)*result = 1;
}

/*
 * isPrimeHybrid
 *
 * A cross between the fine and coarse grained approaches
 * above. Each block takes a number to check for primality
 * and each thread in the block checks a couple of numbers
 * to see if any divide the target number
 *
 * @params
 *      [in] val_offset - Offset to add to the blockId
 *          to get the value being tested for primality
 *      [out] result - Int array for storing results
 *          NOTE - The kernel assumes this array is zeroed,
 *              and sets values to 1 if the associated number
 *              is NOT prime
 */
__global__ void isPrimeHybrid( ull val_offset, int * result )
{
    //Find thread id, +2 to skip testing 0 and 1
    ull tid = threadIdx.x+2;

    //Find the value to be tested for primality
    ull val = blockIdx.x + val_offset;

    //Find the limit to iterate up to (val/2)
    ull lim = val>>1;

    //Assume the value is prime
    bool prime = true;

    //if < 2, not prime
    if(val < 2)
    {
        prime = false;
    }
    else
    {
        //Check multiples of blockDim offset by
        //the thread id to see if any divide the target
        for(ull i=tid; i<lim; i+=blockDim.x)
        {
            if(val % i == 0) prime = false;
        }
    }
    //Merge all results for the warp
    //If any thread in the warp found a divisor
    //Set prime to be false
    prime = !__any(!prime);

    //Every first thread in the warp sets the result
    //Value associated with the blockId to be 1 if any
    //any thread in its warp found a divisor
    if(threadIdx.x%32==0 && !prime) result[blockIdx.x] = 1;
}

/*
 * reduce
 *
 * GPGPU kernel for summing the values in an array
 * in O(lg(n)) time. (The exact time function is 
 * 5*log32(n))
 *
 * @params
 *      [in] data - Pointer to the data to be summed
 *          this data will be modified by the reduction
 *      [in] size - The amount of data to be summed
 *      [in] gapSize - How far apart the data to be
 *          summed sits. The first iteration will be 1,
 *          the second will be 32 and so on
 */
__global__ void reduce( int * data, ull size, ull gapSize )
{
    //Find the thread id
    ull tid = threadIdx.x + blockIdx.x * blockDim.x;
    
    //Find the index to write to
    ull idx = tid*gapSize;

    //Find the first index to read from
    ull offset = 16*gapSize;
    
    if(idx < size)
    {
        //Add the values 16, 8, 4, 2, and 1 above       
        //Unrolled loop for efficiency
        //This works because all threads in the warp
        //Execute in lock-step

        //+16
        if(idx+offset < size)              
            data[idx] += data[idx+offset];
        offset >>= 1;
        
        //+8
        if(idx+offset < size)
            data[idx] += data[idx+offset];
        offset >>= 1;
        
        //+4
        if(idx+offset < size)
            data[idx] += data[idx+offset];
        offset >>= 1;
        
        //+2
        if(idx+offset < size)
            data[idx] += data[idx+offset];
        offset >>= 1;
        
        //+1
        if(idx+offset < size)
            data[idx] += data[idx+offset];
        offset >>= 1;

    }
}

/*
 * getCudaDeviceProperties
 *
 * Returns a string with specific information
 * about the cuda device number 0.
 * Information includes:
 *      Version info
 *      Global memory
 *      # CUDA cores
 *
 * @returns
 *      string of text
 */
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

/*
 * sumRange
 *
 * Uses the reduce kernel to sum the values
 * in a range
 *
 * @params
 *      [in] data - Pointer on device to data to sum
 *         NOTE - This data will be modified 
 *      [in] size - The number of items to sum
 *      [in] warps - The number of warps per block to use
 *
 * @returns
 *      unsigned long long - The sum of the elements of the array
 */
ull sumRange(int* data, ull size, int warps)
{   
    //cout << "\nStarting cuda range sum on " << size << " items" << endl;

    int result;
	ull threadsPer = WARPSIZE*warps;
	
    bool more=false;
    
    //Amount of values each thread owns
    ull threadRange = 1;
    
    //Calculate the number of blocks needed to complete
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
        
        //Run as many blocks as needed for each iteration
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
            
            //Queues a reduce kernel
            reduce<<<blocks, threadsPer>>>(data+blocksDone*threadsPer, size-blocksDone*threadsPer, threadRange);
            blocksDone += blocks;
        }
        //Calculate how many blocks are needed for the next iteration
        threadRange = WARPSIZE*threadRange;
        threadsNeeded = (ull)(ceil((double)size/threadRange)+0.2);
        warpsNeeded = (ull)(ceil((double)threadsNeeded/WARPSIZE)+0.2);
	    blocksNeeded = (ull)(ceil((double)warpsNeeded/warps)+0.2);
        blocksDone = 0;
        
        //Synchronize; we need all queued reduction kernels to finish
        //Before the next iteration
        cudaDeviceSynchronize();

        //Exit if there was an error
        lastError = cudaPeekAtLastError();
        if(lastError != cudaSuccess)
        {
            cout << "sumRange::Error during kernel execution: " << cudaGetErrorString(lastError) << endl;
            return 0;
        }

    //Repeat until we get down to 1 kernel
    }while(more);

    //Copy the item at result[0] and return it
    cudaMemcpy(&result, data, sizeof(int), cudaMemcpyDeviceToHost);
    return result;
}

/*
 * runCudaCoarse
 *
 * Uses the isPrimeCoarse kernel to find the number
 * of primes in a range. Every thread checks to find the 
 * primality of a specific number.
 *
 * @params
 *      [in] start - The beginning of the range to check
 *      [in] end - The end of the range to check
 *      [in] warps - The number of warps per block to use
 *
 * @returns
 *      unsigned long long - The number of primes in the range [start, end]
 */
int runCudaCoarse( ull start, ull end, unsigned int warps )
{
	ull range = end-start+1;
	ull size = (range)*sizeof(int);
	
	//Make sure the call didn't specify more threads per block than allowed
	ull threadsPer = warps*WARPSIZE;
	if(threadsPer > MAXTHREADS) threadsPer = MAXTHREADS;
	
	//Calculate the total number of blocks needed
	ull totalBlocks = (ull)(ceil((double)range/threadsPer)+0.2);
	cudaError_t lastError;
	
	//Allocate memory, check for a failure
	int *d_result;
	if(cudaMalloc( &d_result, size )==cudaErrorMemoryAllocation)
	{
	    cout << "Error allocating memory on cuda device" << endl;
	    return 0;
	}
	
	//Set the memory to 0; non-primes will be set to 1
    cudaMemset(d_result, 0, size);
	
    ull threadsThisTime;
    ull totalDone=0;
    ull blocks;
    
    //Do all of the numbers in the range
	while(totalDone < range)
	{
	    //Do as many threads each iteration as possible
	    threadsThisTime = MAXBLOCKS*threadsPer;
	    if(start+threadsThisTime > end) threadsThisTime = end-start+1;
	    
	    //Calculate the number of blocks to use this call
	    blocks = (ull)(ceil(threadsThisTime/(double)threadsPer)+0.2);

        //cout << "IsPrimeCoarse<<<" << blocks << ", " << threadsPer << ">>>(" << totalDone << ", " << start << ", " << end << ", " << d_result << ")" << endl;
		isPrimeCoarse<<< blocks , threadsPer >>>( start, end, d_result+totalDone ); 
		    
	    //Increment offsets
		start += threadsThisTime;
		totalDone += threadsThisTime;
	}

    //Wait for all cuda kernels to end
    cudaDeviceSynchronize();
    
    //Make sure none of the kernels caused errors
    lastError = cudaPeekAtLastError();
    if(lastError != cudaSuccess)
    {
        cout << "runCudaCoarse::Error during kernel execution: " << cudaGetErrorString(lastError) << endl;
        cudaFree( d_result );
        return 0;
    }
    
    //Get the number of primes
    int count = range-sumRange(d_result, range, warps);

	cudaFree( d_result );
	
	return count;
}

/*
 * runCudaFine
 *
 * Uses the isPrimeFine kernel to find the number
 * of primes in a range. Every thread in the kernel
 * checks if a specific number is divisible by the threadid
 *
 * @params
 *      [in] start - The beginning of the range to check
 *      [in] end - The end of the range to check
 *      [in] warps - The number of warps per block to use
 *
 * @returns
 *      unsigned long long - The number of primes in the range [start, end]
 */
int runCudaFine( ull start, ull end, unsigned int warps )
{
    ull range = end-start+1;
    ull size = (range)*sizeof(int);
    ull threadsPer = warps*WARPSIZE;
    
    //Make sure the call doesn't specify more threads per block
    //than allowed
    if(threadsPer > MAXTHREADS) threadsPer = MAXTHREADS;
	cudaError_t lastError;
    	
	//Allocate memory for result
    int* d_result;
	if(cudaMalloc( &d_result, size )==cudaErrorMemoryAllocation)
	{
	    cout << "Error allocating memory on cuda device" << endl;
	    return 0;
	}
	
	//Set result to 0, non-primes will be set to 1
    cudaMemset(d_result, 0, size);
           
    ull blocks;
    //i - index of return location
    //j - value to test
    int* i = d_result;
    for(ull j=start; j<=end; i++, j++)
    {
        //If the current value being tested is > the max number
        //of threads for a CUDA launch, do multiple launches
        
        //k - Offset from 0 for thread indices
        for(ull k=0; k<=j; k+= threadsPer*MAXBLOCKS)
        {
            //Calculate how many blocks to launch
            blocks = (ull)(ceil((double)(j/2)/threadsPer)+0.2);
            if(blocks > MAXBLOCKS) blocks = MAXBLOCKS;
            
            //Special case for 0
            if(!blocks) blocks = 1;

            //cout << "IsPrimeFine<<<" << blocks << ", " << threadsPer << ">>>(" << j << ", " << k << ", " << i << ")" << endl;
            isPrimeFine<<<blocks, threadsPer>>>( j, k, i );
        }
    }

    //Wait for all cuda kernels to end
    cudaDeviceSynchronize();
    
    //Check if any kernels caused errors
    lastError = cudaPeekAtLastError();
    if(lastError != cudaSuccess)
    {
        cout << "runCudaFine::Error during kernel execution: " << cudaGetErrorString(lastError) << endl;
        cudaFree( d_result );
	    
        return 0;
    }
    
    //Get the number of primes
    int count = range-sumRange(d_result, range, warps);
    
    cudaFree( d_result );
     
    return count;
}

/*
 * runCudaHybrid
 *
 * Uses the isPrimeHybrid kernel to find the number
 * of primes in a range. Every block checks if a different
 * number is prime by having all threads in the block check if
 * a couple of numbers divide it.
 *
 * @params
 *      [in] start - The beginning of the range to check
 *      [in] end - The end of the range to check
 *      [in] warps - The number of warps per block to use
 *
 * @returns
 *      unsigned long long - The number of primes in the range [start, end]
 */
int runCudaHybrid( ull start, ull end, unsigned int warps )
{
    ull range = end-start+1;
    ull size = (range)*sizeof(int);
    ull threadsPer = warps*WARPSIZE;
    
    //Make sure the call doesn't specify more threads in
    //a block than allowed
    if(threadsPer > MAXTHREADS) threadsPer = MAXTHREADS;
	cudaError_t lastError;
	
	//Allocate memory for result
    int* d_result;
	if(cudaMalloc( &d_result, size )==cudaErrorMemoryAllocation)
	{
	    cout << "Error allocating memory on cuda device" << endl;
	    return 0;
	}
	
	//Set result to 0; non-primes will be set to 1
    cudaMemset(d_result, 0, size);
    
    ull blocks;
    //Need to run a total of 'range' blocks
    //If that's more than 2^16, then this loop will
    //run more than once
    for(ull i=start, j=0; i<end; i+= MAXBLOCKS, j+=MAXBLOCKS)
    {
        blocks = end-i+1;
        if(blocks > MAXBLOCKS) blocks = MAXBLOCKS;
        
        //cout << "IsPrimeHybrid<<<" << blocks << ", " << threadsPer << ">>>(" << j << ", " << i << ", " << d_result << ")" << endl;
        isPrimeHybrid<<<blocks, threadsPer>>>( i, d_result+j);
    }

    //Wait for all kernel calls to end
    cudaDeviceSynchronize();
    
    //Check if any kernels caused errors
    lastError = cudaPeekAtLastError();
    if(lastError != cudaSuccess)
    {
        cout << "runCudaHybrid::Error during kernel execution: " << cudaGetErrorString(lastError) << endl;
        cudaFree( d_result );
	    
        return 0;
    }
    
    //Get the number of primes
    int count = range-sumRange(d_result, range, warps);
    
    cudaFree( d_result );
       
    return count;
}
