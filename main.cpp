#include <iostream>
#include <map>
#include <string>
#include <functional>
#include <cstdlib>

#define SEQUENTIAL_NAME "Sequential"
#include "functions.h"
#include "asyncPrime.h"
#include "ompPrime.h"
#include "sequentialPrime.h"

#ifdef _CUDA_PRIME
#include "cudaPrime.cuh"
#endif

using namespace std;

int main( int argc, char** argv )
{
    if( argc != 3 && argc != 5 )
    {
        cout << "Usages: " << endl;
        cout << "primes start end [increment iterations]\n\tFind the number of primes between start and end. If increment and iterations are given, the program will run through every multiple of increment between start and end the number of times defined by iterations and output the average timings" << endl;
        cout << "Examples:\n\tprimes 10000001 10001000\n\tprimes 0 1000 10 1000" << endl;
        return -1;
    }
    
    map<string, primesFunction> tests;
    
    using namespace placeholders;
    tests[SEQUENTIAL_NAME] = runSequential;
    tests["Omp (Static, 1)"] = bind(runOmpStatic, _1, _2, 1);
    tests["Omp (Static, 10)"] = bind(runOmpStatic, _1, _2, 10);
    tests["Omp (Dynamic, 1)"] = bind(runOmpDynamic, _1, _2, 1);
    tests["Omp (Dynamic, 10)"] = bind(runOmpDynamic, _1, _2, 10);
    tests["std::Async"] = runAsync;
    
#ifdef _CUDA_PRIME
    tests["Cuda - Coarse: 1 warp"] = bind(runCudaCoarse, _1, _2, 1);
    tests["Cuda - Fine: 1 warp"] = bind(runCudaFine, _1, _2, 1);
    tests["Cuda - Hybrid: 1 warp"] = bind(runCudaHybrid, _1, _2, 1);
    tests["Cuda - Coarse: 16 warps"] = bind(runCudaCoarse, _1, _2, 16);
    tests["Cuda - Fine: 16 warps"] = bind(runCudaFine, _1, _2, 16);
    tests["Cuda - Hybrid: 16 warps"] = bind(runCudaHybrid, _1, _2, 16);
    tests["Cuda - Coarse: 32 warps"] = bind(runCudaCoarse, _1, _2, 32);
    tests["Cuda - Fine: 32 warps"] = bind(runCudaFine, _1, _2, 32);
    tests["Cuda - Hybrid: 32 warps"] = bind(runCudaHybrid, _1, _2, 32);
#endif
    
    unsigned long long start, end, inc, iter;
    start = strtoull(argv[1], NULL, 10);
    end = strtoull(argv[2], NULL, 10);
    if(argc == 3)
    {
        findNumPrimes(tests, start, end);
    }
    else
    {
        inc = strtoull(argv[3], NULL, 10);
        iter = strtoull(argv[4], NULL, 10);
        getTimeStats(tests, start, end, inc, iter);
    }
        
    return 0;
}

