#include "ompPrime.h"
#include "functions.h"
#include <omp.h>
#include <iostream>

using namespace std;

/*
 * runOmpStatic
 *
 * Runs the prime function in paralell using
 * omp and static scheduling. This means that
 * every thread started will do the same amount
 * of work, which will be doled out in a round
 * robin fashion, in groups of size defined by
 * the parameter.
 *
 * @params
 *      [in] start - The start of the range to
 *          find primes in
 *      [in] end - The end of the range to find
 *          primes in
 *      [in] work - The size of work groups
 *          to hand out to threads
 *
 *  @returns
 *      int - The number of primes found in the
 *          range [start, end]
 */
int runOmpStatic( ull start, ull end, ull work )
{
    int total = 0;
    //Omp parallel for
    //Uses as many threads as the cpu reports having
    //Uses static scheduling with 'work' sized groups
    //Add reduction on 'total' allows thread-safe summing
    //Of results
#   pragma omp parallel for num_threads(omp_get_num_procs()) schedule(dynamic, work) \
    reduction(+ : total)
    for(ull i = start; i <= end; i++)
    {
        if(isPrime(i)) total+=1;
    }
    
    return total;
}

/*
 * runOmpDynamic
 *
 * Runs the prime function in paralell using
 * omp and dynamic scheduling. This means that
 * work will be handed out to which ever thread
 * is free, in groups of size defined by the parameter.
 *
 * @params
 *      [in] start - The start of the range to
 *          find primes in
 *      [in] end - The end of the range to find
 *          primes in
 *      [in] work - The size of work groups
 *          to hand out to threads
 *
 * @returns
 *      int - The number of primes found in the
 *          range [start, end]
 */

int runOmpDynamic( ull start, ull end, ull work )
{
    int total = 0;
    //Omp parallel for
    //Uses as many threads as the cpu reports having
    //Uses dynamic scheduling with 'work' sized groups
    //Add reduction on 'total' allows thread-safe summing
    //Of results
#   pragma omp parallel for num_threads(omp_get_num_procs()) schedule(dynamic, work) \
    reduction(+ : total)
    for(ull i = start; i <= end; i++)
    {
        if(isPrime(i)) total+=1;
    }
    
    return total;
}
