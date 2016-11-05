#ifndef _FUNCTIONS_H
#define _FUNCTIONS_H

#ifndef SEQUENTIAL_NAME
#define SEQUENTIAL_NAME "Sequential"
#endif

#include <functional>
#include <map>
#include <string>
#include <vector>
#include <iostream>

typedef unsigned long long ull;
typedef std::function<int(ull, ull)> primesFunction;

double timePrimesFunction(primesFunction primeFunc, ull start, ull end, ull& primes);
bool isPrime( ull x );
void getPrimeTimings(std::map<std::string, primesFunction>& tests, ull start, ull end, std::map<std::string, double>& times, std::map<std::string, ull>& primes);
void getTimeAvg(std::map<std::string, primesFunction>& tests, ull start, ull end, ull iterations, std::map<std::string, double>& avgs, std::map<std::string, ull>& primes);
void outputDataHeader(std::ostream& out);
void outputData(std::ostream& out, std::map<std::string, primesFunction>& tests, std::map<std::string, double>& times, std::map<std::string, ull>& primes);
#endif
