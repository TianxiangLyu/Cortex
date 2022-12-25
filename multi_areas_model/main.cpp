#include <iostream>
#include <cortex.hpp>
#include "exc.hpp"
int main(int argc, char *argv[])
{
    CX::Initialize(argc, argv);
    
    CX::Finalize();
    return 0;
}
