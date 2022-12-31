/* #include <iostream>
#include <cortex.hpp> */
#include <neuron/iaf_psc_exp.hpp>
/* namespace V1
{
    CX::Population<iaf_psc_exp<>>::Default L1E;
    CX::Population<iaf_psc_exp<>>::Default L1I;
    void init()
    {
        MPI_Group world_group;
        MPI_Comm_group(MPI_COMM_WORLD, &world_group);
        L1E.initialize("V1_L1e", CX::BOUNDARY_CONDITION_NULL, DistrEqualNullPos(18000), world_group);
        L1I.initialize("V1_L1i", CX::BOUNDARY_CONDITION_NULL, DistrEqualNullPos(4500), world_group);
    }
} */
