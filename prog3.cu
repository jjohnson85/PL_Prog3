#include<iostream>

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

}


