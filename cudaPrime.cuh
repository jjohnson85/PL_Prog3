
#include<iostream>
#include "functions.h"

int runCudaCoarse( ull start, ull end, unsigned int warps );
int runCudaFine( ull start, ull end, unsigned int warps );
int runCudaHybrid( ull start, ull end, unsigned int warps );
std::string getCudaDeviceProperties();
