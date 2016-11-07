
#include "asyncPrime.h"
#include "functions.h"
#include<future>
#include<vector>


/*
 * runAsync
 *
 * Takes unsigned long long start, unsigned long long end, and int batchSize.
 * Asynchronously computes prime numbers from start to end in batches of size
 * batchSize. Returns the number of primes found in the range
 *
 * @params
 *      [in] start - The start of the range to find primes in
 *      [in] end - The end of the range to find primes in
 *	    [in] batchSize - size of async thred batches
 *
 * @returns
 *      int - The number of primes in the range [start, end]
 */
int runAsync( ull start, ull end, int batchSize  )
{
    std::vector<std::future<bool>>futurevector(batchSize);
    int primes = 0;
    int remainingRange = (end-start+1);
    int batchCount = 0;
    int offset;

    while( remainingRange > 0 )
    {

	offset = batchSize * batchCount;
	if( remainingRange < batchSize )
	{
	    batchSize = remainingRange;
	}
	

	for( int i = 0; i < batchSize; i++ )
	{
	    futurevector[i] = async( std::launch::async, isPrime, start+i+offset ); 
	}   

	for( int i = 0; i < batchSize; i++ )
	{
	    futurevector[i].wait( );
	    primes+=int(futurevector[i].get( ));
	}    
	remainingRange -= batchSize;
	batchCount++;
    }
    

    return primes;
}
