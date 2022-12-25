#include <iostream>
#include <cortex.hpp>
#include <neuron/iaf_psc_alpha.hpp>
namespace brunel_exc
{
    struct model_params
    {
        constexpr static const CX::F64 tau_m = 10.0;  // Membrane time constant(ms)
        constexpr static const CX::F64 C_m = 250.0;   // Capacity of the membrane(pF)
        constexpr static const CX::S32 t_ref = 5;     // Duration of refractory period(ms)
        constexpr static const CX::F64 E_L = 0.0;     // Resting membrane potential(mV)
        constexpr static const CX::F64 I_e = 0.0;     // Reset Potential(mV)
        constexpr static const CX::F64 V_reset = 0.0; // mV, rel to E_L
        constexpr static const CX::F64 V_th = 20.0;   // mV, rel to E_L
        constexpr static const CX::F64 LowerBound_ = -std::numeric_limits<CX::F64>::infinity();
        constexpr static const CX::F64 tau_syn_ex = tau_syn; // time const. postsynaptic excitatory currents(ms)
        constexpr static const CX::F64 tau_syn_in = tau_syn; // time const. postsynaptic inhibitory currents(ms)
    };
    constexpr static const CX::F64 tau_syn = 0.32582722403722841;
    typedef iaf_psc_alpha<model_params> iaf_psc;
    const CX::S32 NE = 9000;
    const CX::S32 NI = 2250;
    CX::Population<iaf_psc>::Default L1e;
}
