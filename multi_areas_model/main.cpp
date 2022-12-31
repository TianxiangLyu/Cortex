#include <iostream>
#include <cortex.hpp>
//#include "V1.hpp"
#include <neuron/iaf_psc_exp.hpp>
#include <synapse/stdp_pl_synapse_hom.hpp>
#include <synapse/syn_static.hpp>
#include <synapse/syn_static_hom.hpp>
class stdp_params
{
public:
    constexpr static const CX::F64 delay = 1.5;
    constexpr static const CX::F64 alpha = 0.0513;
    constexpr static const CX::F64 lambda = 0.1;
    constexpr static const CX::F64 mu = 0.4;
    constexpr static const CX::F64 tau_plus = 15.0;
    constexpr static const CX::F64 tau_minus = 30.0; // 30 or 20
};
class syn_params
{
public:
    constexpr static const CX::F64 delay = 1.5;
};
CX::Population<iaf_psc_exp<>>::Default L1e;
CX::Population<iaf_psc_exp<>>::Default L1i;
typedef stdp_pl_synapse_hom<stdp_params> stdp;
typedef syn_static_hom<syn_params> syn;
CX::Connection<iaf_psc_exp<>, stdp, iaf_psc_exp<>> L1e_to_L1e;
CX::Connection<iaf_psc_exp<>, syn, iaf_psc_exp<>> L1e_to_L1i;
CX::Connection<iaf_psc_exp<>, syn, iaf_psc_exp<>> L1i_to_L1e;
CX::Connection<iaf_psc_exp<>, syn, iaf_psc_exp<>> L1i_to_L1i;
int main(int argc, char *argv[])
{
    CX::Initialize(argc, argv);
    CX::Comm::barrier();
    //V1::init();
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    L1e.initialize("V1_L1e", CX::BOUNDARY_CONDITION_NULL, DistrEqualNullPos(18000), world_group);
    L1i.initialize("V1_L1i", CX::BOUNDARY_CONDITION_NULL, DistrEqualNullPos(4500), world_group);
    L1e_to_L1e.initialize(L1e, L1e, iaf_psc_exp<>::Channel::EXC, SetIndegree(9000, Multapses_YES, Autapses_NO), SetWeightFixed(45.61));
    L1e_to_L1i.initialize(L1e, L1i, iaf_psc_exp<>::Channel::EXC, SetIndegree(9000, Multapses_YES, Autapses_YES), SetWeightFixed(45.61));
    L1i_to_L1e.initialize(L1i, L1e, iaf_psc_exp<>::Channel::INH, SetIndegree(2250, Multapses_YES, Autapses_YES), SetWeightFixed(-5.0 * 45.61));
    L1i_to_L1i.initialize(L1i, L1i, iaf_psc_exp<>::Channel::INH, SetIndegree(2250, Multapses_YES, Autapses_NO), SetWeightFixed(-5.0 * 45.61));
    CX::Comm::barrier();
    CX::Finalize();
    return 0;
}
