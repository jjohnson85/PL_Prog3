#include <functional>
#include <vector>
#include <string>
#include <map>
#include <iostream>
#include <iomanip>
#include <omp.h>

#include "functions.h"

using namespace std;

double timePrimesFunction(primesFunction primeFunc, ull start, ull end, ull& primes)
{
    double startTime = omp_get_wtime();
    
    primes = primeFunc(  start, end );
    
    double endTime = omp_get_wtime();

    return (endTime-startTime);
}

bool isPrime( ull x )
{
    if( x < 2 )
       return false;

    bool isPrime = true;
    for( ull i = 2; i < x / 2; i++ )
    {
        if( x % i == 0 )
        {
            isPrime = false;
        }           
    }
    return isPrime;
}

void getPrimeTimings(map<string, primesFunction>& tests, ull start, ull end, map<string, double>& times, map<string, ull>& primes)
{
    times.clear();
    primes.clear();
    
    for(auto t : tests)
    {
        times[t.first] = timePrimesFunction(t.second, start, end, primes[t.first]);
    }
}

void outputData(std::map<std::string, primesFunction>& tests, std::map<string, double>& times, std::map<string, ull>& primes)
{
    int counter = 0;
    int dataWidth = 25;
    double seqTime = times[SEQUENTIAL_NAME];
    int precision = 3;
    
    cout << left << setw(dataWidth) << "Algorithm" << right << setw(dataWidth) << "nprimes" << setw(dataWidth) << "time(sec)" << setw(dataWidth) << "speedup" << endl;
    cout << string(dataWidth * 4, '-') << endl;
    
    cout << left << setw(dataWidth) << SEQUENTIAL_NAME << right << setw(dataWidth) << primes[SEQUENTIAL_NAME] << setw(dataWidth) << fixed << setprecision(precision) << times[SEQUENTIAL_NAME] << setw(dataWidth) << seqTime/times[SEQUENTIAL_NAME] << endl;

    for(auto t : tests)
    {
        if(t.first != SEQUENTIAL_NAME)
        {
            cout << left << setw(dataWidth) << t.first << right << setw(dataWidth) << primes[t.first] << setw(dataWidth) << fixed << setprecision(precision) << times[t.first] << setw(dataWidth) << seqTime/times[t.first] << endl;
        }
        counter++;
    }
}

void findNumPrimes(map<string, primesFunction>& tests, ull start, ull end)
{
    map<string, double> times;
    map<string, ull> primes;
    getPrimeTimings(tests, start, end, times, primes);
    
    outputData(tests, times, primes);
}

void getTimeStats(map<string, primesFunction>& tests, ull start, ull end, ull increment, ull iterations)
{

    map<string, double> times;
    map<string, ull> primes;
    
    int count = 0;
    for(ull i=start; i<=end; i+=increment, count++)
    {
        map<string, double> avgs;
        for(ull j=0; j<iterations; j++)
        {
            getPrimeTimings(tests, start, i, times, primes);
            for(auto t : tests)
                avgs[t.first]+=times[t.first];
        }
        for(auto t : tests)
            avgs[t.first]/=iterations;
            
        cout << "[" << start << ", " << i << "]" << endl;
        outputData(tests, avgs, primes);
        
        cout << endl;
    }
}
