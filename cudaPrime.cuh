
#include<iostream>

typedef unsigned long long ull;

int runCudaCoarse( unsigned long long start, unsigned long long end, unsigned int warps );
int runCudaFine( unsigned long long start, unsigned long long end, unsigned int warps );
int runCudaHybrid( unsigned long long start, unsigned long long end, unsigned int warps );
