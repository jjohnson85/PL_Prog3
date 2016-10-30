#include<iostream>
#include<chrono>

using namepace std;

bool isPrime( unsigned int x )
{
    if( x < 2 )
    {
       return false;
    }
    else
    {
        for( unsigned int i = 2; i < x; i++ )
        {
            if( x % i == 0 )
            {
                return true;
            }           
        }
        return false;
    }
}

int runSequential( unsigned int start, unsigned int end )
{
    int count = 0;

    for( unsigned int i = start; i < end; i++ )
    {
        if( isPrime( i ) )
        {
            count++;
        }        
    }

    return count;
}

int main( int argc, char** argv )
{
    chrono:time_point<chrono:sytem_clock> start, end;

    if( argc > 3 )
    {
        cout << "Usage for this program is: primes [start] [end]" << endl;
        cout << "Example: primes 10000001 10001000" << endl;
        return -1;
    }

    //Run sequential test
    start = chrono::system_clock::now();
    runSequential(  argv[1] , argv[2] );
    end = chrono::system_clock::now( );

    cout << start - end << endl;   
        
    return 0;
}

