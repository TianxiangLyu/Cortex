#include <iostream>
#include <cortex.hpp>
#include <neuron/iaf_psc_alpha.hpp>
#include <synapse/stdp_pl_synapse_hom.hpp>
#include <synapse/syn_static.hpp>
#include <synapse/syn_static_hom.hpp>
#ifdef USE_GSL
#include <gsl/gsl_sf_lambert.h>
#endif
#define OUTPUT_INTERVAL 100
namespace params
{
    constexpr static const CX::S32 nvp = 1;
    constexpr static const CX::F64 scale = 1;
    constexpr static const CX::F64 simtime = 250;   // ms
    constexpr static const CX::F64 presimtime = 50; // ms
    constexpr static const CX::F64 dt = 0.1;
    constexpr static const bool record_spikes = true;
    constexpr static const char path_name[] = ".";
    constexpr static const char log_file[] = "log";
};
#ifdef USE_GSL
CX::F64 ConvertSynapseWeight(const CX::F64 tau_m, const CX::F64 tau_syn, const CX::F64 C_m)
{
    const CX::F64 a = tau_m / tau_syn;
    const CX::F64 b = 1.0 / tau_syn - 1.0 / tau_m;
    const CX::F64 t_rise = 1.0 / b * (-gsl_sf_lambert_Wm1(-exp(-1.0 / a) / a) - 1.0 / a);
    const CX::F64 v_max = exp(1.0) / (tau_syn * C_m * b) * ((exp(-t_rise / tau_m) - exp(-t_rise / tau_syn)) / b - t_rise * exp(-t_rise / tau_syn));
    return 1. / v_max;
}
#endif
constexpr static const CX::F64 tau_syn = 0.32582722403722841;
namespace brunel_params
{
    constexpr static const CX::S32 NE = 9000;
    constexpr static const CX::S32 NI = 2250;
    constexpr static const CX::S32 Nrec = 1000;
    constexpr static const bool randomize_Vm = true;
    constexpr static const CX::F64 mean_potential = 5.7;
    constexpr static const CX::F64 sigma_potential = 7.2;
    constexpr static const CX::F64 delay = 1.5;
    constexpr static const CX::F64 JE = 0.14;
    constexpr static const CX::F64 sigma_w = 3.47;
    constexpr static const CX::F64 g = -5.0;
    constexpr static const CX::F64 eta = 1.685;
    constexpr static const char *filestem = params::path_name;
}
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
template <class Tlayer>
inline void PoissonStimulus(Tlayer &layer,
                            const CX::F64 time, const CX::F64 dt,
                            const CX::F64 rate, const CX::F64 weight)
{
    if (time > 1.5)
    {
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel
#endif
        {
            std::random_device rd;
            std::default_random_engine eng(rd());
            std::poisson_distribution<CX::S64> d(rate * 1e-3 * dt);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
            for (CX::S32 i = 0; i < layer.getNumLocal(); ++i)
                layer[i].input_ex_ += d(eng) * weight;
        }
    }
}
template <class Tlayer>
inline void RecordSpikeAll(Tlayer &layer)
{
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel for
#endif
    for (CX::S32 i = 0; i < layer.getNumLocal(); ++i)
        layer[i].recordSpike();
}
template <class Tlayer>
inline void SetRandomPotential(Tlayer &layer, const CX::F64 mean_potential, const CX::F64 sigma_potential)
{
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel
#endif
    {
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::normal_distribution<CX::F64> d(mean_potential, sigma_potential);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
        for (CX::S32 i = 0; i < layer.getNumLocal(); i++)
            layer[i].y3_ = d(eng);
    }
}
int main(int argc, char *argv[])
{
    CX::Initialize(argc, argv);
    CX::Comm::barrier();
    if (CX::Comm::getRank() == 0)
    {
        std::cout << "===========================================" << std::endl
                  << "Cortex HPC_benchmark " << std::endl
                  << "===========================================" << std::endl
                  << "Num of processes is " << CX::Comm::getNumberOfProc() << std::endl
                  << "Num of thread is    " << CX::Comm::getNumberOfThread() << std::endl
                  << "===========================================" << std::endl;
    }
    CX::Comm::barrier();
    CX::F64 scale = params::scale;
    CX::F64 presimtime = params::presimtime;
    CX::F64 simtime = params::simtime;

    CX::S32 c;
    while ((c = getopt(argc, argv, "z:p:s:")) != -1)
    {
        switch (c)
        {
        case 'z':
            scale = atof(optarg);
            if (CX::Comm::getRank() == 0)
                std::cerr << "scale = " << scale << std::endl;
            break;
        case 'p':
            presimtime = atof(optarg);
            if (CX::Comm::getRank() == 0)
                std::cerr << "presimtime = " << presimtime << std::endl;
            break;
        case 's':
            simtime = atof(optarg);
            if (CX::Comm::getRank() == 0)
                std::cerr << "simtime = " << simtime << std::endl;
            break;
        default:
            if (CX::Comm::getRank() == 0)
                std::cerr << "No such option! Available options are here." << std::endl;
            CX::Abort();
        }
    }
    CX::Comm::barrier();
    const CX::F64 init_offset = CX::GetWtime();
    CX::F64 dt = params::dt;
    CX::F64 end_time = presimtime + simtime; // length means the distance from center to the boundary
    const CX::S32 NE = brunel_params::NE * scale;
    const CX::S32 NI = brunel_params::NI * scale;
    const CX::F64 size_scale = sqrt(scale);
    const CX::S32 n_proc = CX::Comm::getNumberOfProc();

    const CX::S32 CE = 9000;
    const CX::S32 CI = 2250;

#ifdef USE_GSL
    const CX::F64 conversion_factor = ConvertSynapseWeight(model_params::tau_m, model_params::tau_syn_ex, model_params::C_m);
#else
    const CX::F64 conversion_factor = 325.783;
#endif
    const CX::F64 JE_pA = conversion_factor * brunel_params::JE;
    const CX::F64 nu_thresh = model_params::V_th / (CE * model_params::tau_m / model_params::C_m * JE_pA * expf(1.0) * tau_syn);
    const CX::F64 nu_ext = nu_thresh * brunel_params::eta;

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    typedef iaf_psc_alpha<model_params> iaf_psc;
    typedef stdp_pl_synapse_hom<stdp_params> stdp;
    typedef syn_static_hom<syn_params> syn;

    CX::Population<iaf_psc>::Default L1e("L1e", CX::BOUNDARY_CONDITION_NULL, DistrEqualNullPos(NE), world_group);
    CX::Population<iaf_psc>::Default L1i("L1i", CX::BOUNDARY_CONDITION_NULL, DistrEqualNullPos(NI), world_group);
    // Using a specific MPI_Group to determine the layer allocation on specific processes.
    CX::Connection<iaf_psc, stdp, iaf_psc> L1e_to_L1e(L1e, L1e, iaf_psc::Channel::EXC, SetIndegree(CE, Multapses_YES, Autapses_NO), SetWeightFixed(JE_pA));
    CX::Connection<iaf_psc, syn, iaf_psc> L1e_to_L1i(L1e, L1i, iaf_psc::Channel::EXC, SetIndegree(CE, Multapses_YES, Autapses_YES), SetWeightFixed(JE_pA));
    CX::Connection<iaf_psc, syn, iaf_psc> L1i_to_L1e(L1i, L1e, iaf_psc::Channel::INH, SetIndegree(CI, Multapses_YES, Autapses_YES), SetWeightFixed(brunel_params::g * JE_pA));
    CX::Connection<iaf_psc, syn, iaf_psc> L1i_to_L1i(L1i, L1i, iaf_psc::Channel::INH, SetIndegree(CI, Multapses_YES, Autapses_NO), SetWeightFixed(brunel_params::g * JE_pA));

    SetRandomPotential(L1e, brunel_params::mean_potential, brunel_params::sigma_potential);
    SetRandomPotential(L1i, brunel_params::mean_potential, brunel_params::sigma_potential);

    CX::S32 step = 1;
    CX::Comm::barrier();
    if (CX::Comm::getRank() == 0)
        std::cout << "init time " << CX::GetWtime() - init_offset << std::endl;
    const CX::F64 time_offset = CX::GetWtime();
    for (CX::F64 time = 0; time < presimtime; time += dt, step++)
    {
        L1e.SpikeUpdate(time);
        L1i.SpikeUpdate(time);
        L1e_to_L1e.PreAct(time, stdp::CalcInteraction(time));
        L1e_to_L1i.PreAct(time, syn::CalcInteraction(JE_pA));
        L1i_to_L1e.PreAct(time, syn::CalcInteraction(brunel_params::g * JE_pA));
        L1i_to_L1i.PreAct(time, syn::CalcInteraction(brunel_params::g * JE_pA));
        PoissonStimulus(L1e, time, dt, nu_ext * CE * 1000, JE_pA);
        PoissonStimulus(L1i, time, dt, nu_ext * CE * 1000, JE_pA);
        L1e.NeuralDynamics(iaf_psc::CalcDynamics(time, dt));
        L1i.NeuralDynamics(iaf_psc::CalcDynamics(time, dt));
    }
    const CX::F64 pre_sim_time = CX::GetWtime() - time_offset;
    if (CX::Comm::getRank() == 0)
        std::cout << "pre-sim time: " << pre_sim_time << std::endl;
    for (CX::F64 time = presimtime; time < presimtime + simtime; time += dt, step++)
    {
        L1e.SpikeUpdate(time);
        L1i.SpikeUpdate(time);
        L1e_to_L1e.PreAct(time, stdp::CalcInteraction(time));
        L1e_to_L1i.PreAct(time, syn::CalcInteraction(JE_pA));
        L1i_to_L1e.PreAct(time, syn::CalcInteraction(brunel_params::g * JE_pA));
        L1i_to_L1i.PreAct(time, syn::CalcInteraction(brunel_params::g * JE_pA));
        PoissonStimulus(L1e, time, dt, nu_ext * CE * 1000, JE_pA);
        PoissonStimulus(L1i, time, dt, nu_ext * CE * 1000, JE_pA);
        L1e.NeuralDynamics(iaf_psc::CalcDynamics(time, dt));
        L1i.NeuralDynamics(iaf_psc::CalcDynamics(time, dt));
        RecordSpikeAll(L1e);
        RecordSpikeAll(L1i);
    }
    CX::Comm::barrier();
    const CX::F64 sim_time = CX::GetWtime() - time_offset - pre_sim_time;
    if (CX::Comm::getRank() == 0)
        std::cout << "sim time: " << sim_time << std::endl;
    CX::S32 exc_spk_count = 0;
    CX::S32 inh_spk_count = 0;
    for (CX::S32 i = 0; i < L1e.getNumLocal(); i++)
        exc_spk_count += L1e[i].spikeNum;
    for (CX::S32 i = 0; i < L1i.getNumLocal(); i++)
        inh_spk_count += L1i[i].spikeNum;
    exc_spk_count = CX::Comm::getSum(exc_spk_count);
    inh_spk_count = CX::Comm::getSum(inh_spk_count);
    const CX::F64 avg_spk_exc = exc_spk_count / (CX::F64)L1e.getNumGlobal();
    const CX::F64 avg_spk_inh = inh_spk_count / (CX::F64)L1i.getNumGlobal();
    if (CX::Comm::getRank() == 0)
        std::cout << avg_spk_exc / simtime * 1e3 << " " << avg_spk_inh / simtime * 1e3 << std::endl;
    CX::Finalize();
    return 0;
}
