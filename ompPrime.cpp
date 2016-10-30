#include "ompPrime.h"
#include "functions.h"
#include <omp.h>

int runOmp( ull start, ull end )
{
    int total = 0;
#   pragma omp parallel for num_threads(omp_get_num_procs()) schedule(dynamic, 5) \
    reduction(+ : total)
    for(ull i = start; i <= end; i++)
    {
        if(isPrime(i)) total+=1;
    }
    
    return total;
}
