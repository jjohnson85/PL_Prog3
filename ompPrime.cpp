#include "ompPrime.h"
#include "functions.h"
#include <omp.h>
#include <iostream>

using namespace std;

int runOmpStatic( ull start, ull end, ull work )
{
    int total = 0;
#   pragma omp parallel for num_threads(omp_get_num_procs()) schedule(dynamic, work) \
    reduction(+ : total)
    for(ull i = start; i <= end; i++)
    {
        if(isPrime(i)) total+=1;
    }
    
    return total;
}

int runOmpDynamic( ull start, ull end, ull work )
{
    int total = 0;
#   pragma omp parallel for num_threads(omp_get_num_procs()) schedule(dynamic, work) \
    reduction(+ : total)
    for(ull i = start; i <= end; i++)
    {
        if(isPrime(i)) total+=1;
    }
    
    return total;
}
