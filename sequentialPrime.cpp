#include "sequentialPrime.h"
#include "functions.h"

int runSequential( unsigned int start, unsigned int end )
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
