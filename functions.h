#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

#ifndef SEQUENTIAL_NAME
#define SEQUENTIAL_NAME "Sequential"
#endif

#include <functional>
#include <map>
#include <string>
#include <vector>

typedef std::function<int(unsigned int, unsigned int)> primesFunction;
typedef unsigned long long ull;

double timePrimesFunction(primesFunction primeFunc, ull start, ull end, ull& primes);
bool isPrime( ull x );
void getPrimeTimings(std::map<std::string, primesFunction>& tests, ull start, ull end, std::map<std::string, double>& times, std::map<std::string, ull>& primes);
void findNumPrimes(std::map<std::string, primesFunction>& tests, ull start, ull end);
void getTimeStats(std::map<std::string, primesFunction>& tests, ull start, ull end, ull increment, ull iterations);
void outputDataHeader();
void outputData(std::map<std::string, primesFunction>& tests, std::map<std::string, double>& times, std::map<std::string, ull>& primes);
#endif
