/*
 * This program will determine the primality of a range of integersby trial division.
 * Range limits will begiven as command-line arguments.
 * It will also run benchmarks sequentially, in parallel  using  multi-core  CPUs
 * with async and  OpenMP, and  in  parallel  using  GPGPU  with CUDA.
 * The final output lists how many prime numbers were found in the range, how long the processing 
 * took (in seconds), and how much speedup was obtained over sequential processing.
 *
 * There are a variety of functions used to parallelize this algorithm; the
 * CUDA GPGPU algorithms only run if the macro flag _CUDA_PRIME is set at compile
 * time.
 *  * Sequential - Checks numbers for primality sequentially
 *  
 *  * Omp (Static, n) - Checks numbers in parallel using omp static scheduling, assigning
 *      work in groups of n
 *  
 *  * Omp (Dynamic, n) - Checks numbers in parallel using omp dynamic scheduling, assigning
 *      work in groups of n
 *  
 *  * Async (Batch Size n) - Uses the std::Async framework to check primality in parallel.
 *      Because of memory limitations, for large ranges it is not possible to create a
 *      future object for each item in the range, so we limit ourselves to running only n
 *      async operations at any given time.
 *  
 *  * Cuda : Fine Grained; n warp/block - Sequentially checks if each number is prime.
 *      For any number n being checked for primality, each CUDA thread checks if it is
 *      divisible by exactly one of the numbers less than n/2. Blocks are run with n warps
 *  
 *  * Cuda : Coarse Grained; n warp/block - All values are checked for primality in parallel.
 *      Each CUDA thread takes a number, and then uses the given primality check to determine
 *      if that value is parallel. Blocks are run with n warps
 *
 *  * Cuda : Hybrid; n warp/block - All values are checked for primality in parallel.
 *      Each CUDA block takes a number to check for primality, and then each thread in that
 *      block checks multiples of the block dimension offset by the thread id to see if the
 *      value is divisble by that number
 *
 * Compiling:
 *  The openmp library and g++ with c++11 are required
 *  Compile using the included makefile and the following make commands
 *      make cuda - Makes a version of the program which uses CUDA programming. Requires NVCC
 *      make nocuda - Makes a version of the program which does not use CUDA
 *      make - See make cuda
 *
 *      make cudaGraph and make nocudaGraph reformat the output to only a header and
 *          the timing data, allowing for easy use with a graphing tool such as excel or gnuplot
 * 
 * Usage:
 *  ./primes start end [iterations] [increment] [-o] [-a] [-c]
 *      start, end - Defines the range to find primes in, inclusive
 *      iterations - The number of times to run the prime tests and average
 *          runtimes
 *      increment - Defines a step value to increase the range from start to end
 *          for producing graphable time data
 *
 *      By default, the program will only run a subset of the full test suite. The following
 *      flags will run all defined tests for...
 *          -o - OMP
 *          -a - std::Async
 *          -c - CUDA
 *
 *      The defualt algorithm subset includes
 *          Sequential
 *          CUDA : Hybrid, 16
 *          OMP (Static, 10)
 *          Async (Batch Size 32)
 *
 *  Examples:
 *      ./primes 0 10 - Finds the number of primes between 0 and 10
 *      ./primes 0 10 100 - Finds the number of primes between 0 and 10,
 *          runs 100 times and averages the timing
 *      ./primes 0 1000 100 10 - Finds all of the primes in [0, 10], then [0, 20]
 *          then [0, 30] up to [0, 1000]. Each range is run 100 times and the timings
 *          are averaged
 */

#include <iostream>
#include <map>
#include <string>
#include <functional>
#include <cstdlib>
#include <cctype>

//Defines the name that the sequential function
//Will be mapped to so it can be displayed first
#define SEQUENTIAL_NAME "Sequential"
#include "functions.h"
#include "asyncPrime.h"
#include "ompPrime.h"
#include "sequentialPrime.h"

//Only include the cuda header if the compiler
//sets this flag
#ifdef _CUDA_PRIME
#include "cudaPrime.cuh"
#endif

using namespace std;

/*
 * usage
 *
 * Outputs a usage message
 */
void usage()
{
        cout << "Usages: " << endl;
        cout << "primes start end [iterations] [increment] [-o] [-a] [-c]\n\tFind the number of primes between start and end. If increment and iterations are given, the program will run through every multiple of increment between start and end the number of times defined by iterations and output the average timings" << endl;
        cout << "Examples:\n\tprimes 0 1000 -- Find the number of primes between 0 and 1000\n\t" <<
        "primes 0 1000 100 -- Find the number of primes between 0 and 1000, 100 times, and average the times taken\n\t" <<
        "primes 0 1000 100 10 -- Find the number of primes between 0 and 10, then 0 and 20... up to 0 and 1000, doing each range 100 times and averaging the times" << endl;
        cout <<  "By default, the program will only run a subset of the full test suite. The following" << 
        " flags will run all defined tests for...\n" << 
        "\t-o - OMP\n\t-a - std::Async\n\t-c - CUDA\n\n" << 
        "The defualt algorithm subset includes\n\tSequential\n\tCUDA : Hybrid, 16\n\tOMP (Static, 10)\n\tAsync (Batch Size 32)" << endl;
}

/*
 * main
 *
 * Parses command line parameters and determines run mode.
 * There are three modes
 *  * Find the number of primes between some a and b(inclusive)
 *  * Find the number of primes between some a and b(inclusive)
 *      n times, and average the timing data
 *  * Find the number of primes between some a and a+i, a+2i, a+3i...b
 *    (inclusive). Each chunk is done n times and averaged
 *
 * Puts all of the parallel functions into a map of string -> function,
 * and passes that map to the function which does the correct run mode.
 *
 * @params
 *      argc - Number of command line args
 *      argv - Actual command line args
 *
 * @returns
 *      0 - Successful run
 *      1 - Invalid number of command line args
 *      2 - Unparsable command line args
 */
int main( int argc, char** argv )
{
    //Make sure the command line arguements are
    //valid
    if( argc < 3 || argc > 8 )
    {
        usage();
        return 1;
    }
    
    //Parse the start and end arguments
    int intsFound = 0;
    vector<ull> values(4, 0);
    map<char, bool> flags;
    flags['c'] = flags['o'] = flags['a'] = false;
    
    for(int i=1; i<argc; i++)
    {
        //If arg starts with -, flag the chars after it true
        if(argv[i][0] == '-')
        {
            string arg = argv[i];
            for(int j=1; j<arg.size(); j++)
            {
                flags[tolower(arg[j])] = true;
            }
        }
        else
        {
            //Otherwise, assume it's a number in the order defined
            values[intsFound++] = (max(strtoll(argv[i], NULL, 10), (long long)0));
        }
    }
    
    if(intsFound > 4 || intsFound < 2)
    {
        usage();
        return 2;
    }
    
    //Rename the args for convenience
    unsigned long long start=values[0], end=values[1], iter=values[2], inc=values[3];
    
    //Set up the map to pass to the timing functions
    map<string, primesFunction> tests;
    
    //Use placeholders for the std::bind function
    using namespace placeholders;
    
    //Add the sequential function
    tests[SEQUENTIAL_NAME] = runSequential;
    
    //Add the omp functions with 1 and 10 work items
    if(flags['o'])
    {
        tests["Omp (Static, 1)"] = bind(runOmpStatic, _1, _2, 1);
        tests["Omp (Static, 10)"] = bind(runOmpStatic, _1, _2, 10);
        tests["Omp (Dynamic, 1)"] = bind(runOmpDynamic, _1, _2, 1);
    }
    tests["Omp (Dynamic, 10)"] = bind(runOmpDynamic, _1, _2, 10);
    
    //Add the async function
    if(flags['a'])
    {
        tests["Async (Batch Size 4)"] = bind(runAsync, _1, _2, 4);
        tests["Async (Batch Size 8)"] = bind(runAsync, _1, _2, 8);
        tests["Async (Batch Size 16)"] = bind(runAsync, _1, _2, 16);
    }
    tests["Async (Batch Size 32)"] = bind(runAsync, _1, _2, 32);
    
#ifdef _CUDA_PRIME
    //Add the cuda functions, all with 1, 16, and 32 warps
    if(flags['c'])
    {
        tests["Cuda - Coarse: 1 warp"] = bind(runCudaCoarse, _1, _2, 1);
        tests["Cuda - Fine: 1 warp"] = bind(runCudaFine, _1, _2, 1);
        tests["Cuda - Hybrid: 1 warp"] = bind(runCudaHybrid, _1, _2, 1);
        tests["Cuda - Coarse: 16 warps"] = bind(runCudaCoarse, _1, _2, 16);
        tests["Cuda - Fine: 16 warps"] = bind(runCudaFine, _1, _2, 16);
        tests["Cuda - Coarse: 32 warps"] = bind(runCudaCoarse, _1, _2, 32);
        tests["Cuda - Fine: 32 warps"] = bind(runCudaFine, _1, _2, 32);
        tests["Cuda - Hybrid: 32 warps"] = bind(runCudaHybrid, _1, _2, 32);
    }
    tests["Cuda - Hybrid: 16 warps"] = bind(runCudaHybrid, _1, _2, 16);
#endif
    
    
    //Output the data header
    cout << endl;
    outputDataHeader(cout);
    cout << endl;
   
    //If GRAPHDATA defined, output headers in a format
    //which is easy to import into excel
#ifdef GRAPHDATA
    cout << "\"Range\" \"" << SEQUENTIAL_NAME << "\" ";
    for(auto i : tests)
        if(i.first != SEQUENTIAL_NAME)
            cout << "\"" << i.first << "\" ";
    cout << endl;
#else
    cout << "Primes in the range [" << start << ", " << end << "]. Range of " << end-start+1 << " integers" << endl;
    if(intsFound > 2)
        cout << iter << " iterations" << endl;
    if(intsFound > 3)
        cout << "Range increases by " << inc << endl;
#endif

    //Maps to store the times recorded and numbers
    //of primes found
    map<string, double> times;
    map<string, ull> primes;
    
    //If 3 args, run each test 1 time
    if(intsFound == 2)
    {
        getPrimeTimings(tests, start, end, times, primes);
        outputData(cout, tests, times, primes);
    }
    //If 4 args, run each test 'iter' times (arg #3)
    else if(intsFound == 3)
    {
        getTimeAvg(tests, start, end, iter, times, primes);
        outputData(cout, tests, times, primes);
    }
    //Otherwise, run tests from start->start+n*increment up to start->end
    //To get timings over a range of range sizses
    else if(intsFound == 4)
    {
        for(ull i=start; i<=end; i+= inc)
        {
            getTimeAvg(tests, start, i, iter, times, primes);
            
            //If GRAPHDATA defined, output differently for the
            //data x-axis
#ifndef GRAPHDATA
            cout << "Range: [" << start << ", " << i << "]" << endl;
#else
            cout << i-start << " ";
#endif
            outputData(cout, tests, times, primes);
            cout << endl;
        }
    }
        
    return 0;
}

