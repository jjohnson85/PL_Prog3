#include "sequentialPrime.h"
#include "functions.h"

/*
 * runSequential
 *
 * Finds the number of primes in a range. Each number
 * in the range is checked for primality using the isprime
 * function.
 *
 * @params
 *      [in] start - The start of the range to check
 *      [in] end - The end of the range to check
 *
 * @returns
 *      int - The number of primes in the range [start, end]
 */
int runSequential( ull start, ull end )
{
    int count = 0;

    for( unsigned int i = start; i <= end; i++ )
    {
        if( isPrime( i ) )
        {
            count++;
        }        
    }

    return count;
}
