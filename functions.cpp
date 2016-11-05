#include <functional>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>
#include <omp.h>
#include <ctime>
#include "functions.h"

#ifdef _CUDA_PRIME
#include "cudaPrime.cuh"
#endif


using namespace std;

/*
 * timePrimesFunction
 *
 * Takes a primesFunction type, which is defined as a function<int(ull, ull)>
 * and runs it, timing the runtime and getting the number of primes found
 * This number of primes is returned by a reference parameter, and the function
 * returns the number of seconds passed in seconds.
 *
 * @params
 *      [in] primeFunc - The function to time
 *      [in] start - The start of the range to find primes in
 *      [in] end - The end of the range to find primes in
 *      [out] primes - The number of primes the function finds
 *
 * @returns
 *      double - The time the function took in seconds
 */
double timePrimesFunction(primesFunction primeFunc, ull start, ull end, ull& primes)
{
    //Get omp walltime for start
    double startTime = omp_get_wtime();
    
    //Run the function
    primes = primeFunc(  start, end );
    
    //Get omp walltime for end
    double endTime = omp_get_wtime();

    //Return time difference
    return (endTime-startTime);
}

/*
 * isPrime
 *
 * Given function to test is an unsigned long long is prime.
 * Uses an iterative test from 0-n/2, checking if any numbers
 * divide n
 *
 * @params
 *      [in] x - The number to test for primality
 *
 * @returns
 *      bool - true if x is prime, false if not
 */
bool isPrime( ull x )
{
    //If x < 2, not prime
    if( x < 2 )
       return false;

    //Loop from 2-x/2
    bool isPrime = true;
    for( ull i = 2; i < x / 2; i++ )
    {
        //Switch flag false if any number
        //Divides x
        if( x % i == 0 )
        {
            isPrime = false;
        }           
    }
    
    //Return final result
    return isPrime;
}

/*
 * getPrimeTimings
 *
 * Takes a map of string->primeFunction, start and end values, and
 * references to maps to output results. Runs each primeFunction
 * through the timePrimesFunction and stores the number of primes found
 * and the time taken.
 *
 * @params
 *      [in] tests - A map of string name to functions which check for primality and should be timed
 *      [in] start - The start of the range to find primes in
 *      [in] end - The end of the range to find primes in
 *      [out] times - A map of string name to how much time a primesFunction took
 *      [out] primes - A map of string name to how many primes a primesFunction found
 */
void getPrimeTimings(map<string, primesFunction>& tests, ull start, ull end, map<string, double>& times, map<string, ull>& primes)
{
    //Clear the return maps
    times.clear();
    primes.clear();
    
    //Run each test, storing results by the same names
    for(auto t : tests)
    {
        times[t.first] = timePrimesFunction(t.second, start, end, primes[t.first]);
    }
}


/*
 * outputDataHeader
 *
 * Outputs sytem information and the current local time
 *
 * @params
 *      [in/out] out - The output stream to write to
 */
void outputDataHeader(ostream& out)
{
    //Get the current local time
    time_t rawtime;
    struct tm * timeinfo;

    time (&rawtime);
    timeinfo = localtime (&rawtime);

    //Output time, number of processes
    out << "Prime benchmark, run " << asctime(timeinfo) << endl;
    out << "CPU: " << omp_get_num_procs() << " hardware threads" << endl;
    
    //If gpu available, output info for it
#ifdef _CUDA_PRIME
    out << "GPGPU: " << getCudaDeviceProperties() << endl;
#endif
}

/*
 * outputData
 *
 * Output a chart containing information about the timings of prime functions.
 * The output contains one line for each prime function in the input map, and
 * the outputted data is the number of primes found by the function, runtime (in seconds)
 * and speedup compared to the sequential function
 *
 * @params
 *      [in\out] out - The output stream to write to
 *      [in] tests - A map of string names to functions
 *      [in] times - A map of string names to timings
 *      [in] primes - A map of string names to primes found
 */
 
void outputData(ostream& out, std::map<std::string, primesFunction>& tests, std::map<string, double>& times, std::map<string, ull>& primes)
{
    int counter = 0;
    int nameWidth = 25;
    int dataWidth = 15;
    double seqTime = times[SEQUENTIAL_NAME];
    int precision = 10;

//Output a different header to the table if GRAPHDATA defined
//Always output the sequential information first
#ifndef GRAPHDATA    
    out << left << setw(nameWidth) << "Algorithm" << right << setw(dataWidth) << "nprimes" << setw(dataWidth) << "time(sec)" << setw(dataWidth) << "speedup" << endl;
    out << string(dataWidth * 3+nameWidth, '-') << endl;

    out << left << setw(nameWidth) << SEQUENTIAL_NAME << right << setw(dataWidth) << primes[SEQUENTIAL_NAME] << setw(dataWidth) << fixed << setprecision(precision) << times[SEQUENTIAL_NAME] << setw(dataWidth) << seqTime/times[SEQUENTIAL_NAME] << endl;
#else
    out << setprecision(precision) << fixed << right << setw(dataWidth) << times[SEQUENTIAL_NAME] << " ";
#endif

    //Output all of the other data
    for(auto t : tests)
    {
        if(t.first != SEQUENTIAL_NAME)
        {
            //Output only the time data if GRAPHDATA defined
#ifndef GRAPHDATA
            out << left << setw(nameWidth) << t.first << right << setw(dataWidth) << primes[t.first] << setw(dataWidth) << fixed << setprecision(precision) << times[t.first] << setw(dataWidth) << seqTime/times[t.first] << endl;
#else
            out << right << setw(dataWidth) << times[t.first] << " ";
#endif
        }
        counter++;
    }
}

/*
 * getTimeAvg
 *
 * Runs primeFunctions repeatedly and averages the times taken
 * returns the number of primes found on the final iteration
 *
 * @params
 *      [in] tests - Map of string names to functions
 *      [in] start - Start of the range to test for primes
 *      [in] end - End of the range to test for primes
 *      [in] iterations - Number of times to run each function for the average
 *      [out] avgs - Map of string names to average times taken
 *      [out] primes - Map of string names to number of primes found on final iteration
 */
void getTimeAvg(map<string, primesFunction>& tests, ull start, ull end, ull iterations, map<string, double>& avgs, map<string, ull>& primes)
{
    //Set up temporary map of timings
    map<string, double> times;
    
    //Zero avgs map
    for(auto t : tests)
        avgs[t.first] = 0;
        
    //Call getPrimeTimings a bunch of times
    for(ull i=0; i<iterations; i++)
    {
        getPrimeTimings(tests, start, end, times, primes);
        
        //Sum the timings
        for(auto t : tests)
            avgs[t.first] += times[t.first];
    }
    
    //Divide the timings to get average
    for(auto t : tests)
        avgs[t.first] /= iterations;
}
