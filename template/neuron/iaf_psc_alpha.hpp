#include <cortex.hpp>
#include <limits>
#include <stdlib.h>
#include <cortex_utils.hpp>
namespace CX = Cortex;
struct iaf_psc_alpha_default
{
    // neuron
    constexpr static const CX::F64 tau_m = 10.0;          // Membrane time constant(ms)
    constexpr static const CX::F64 C_m = 250.0;           // Capacity of the membrane(pF)
    constexpr static const CX::F64 t_ref = 2.0;           // Duration of refractory period(ms)
    constexpr static const CX::F64 E_L = -70.0;           // Resting membrane potential(mV)
    constexpr static const CX::F64 I_e = 0.0;             // Reset Potential(mV)
    constexpr static const CX::F64 V_reset = -70.0 - E_L; // mV, rel to E_L
    constexpr static const CX::F64 V_th = -55.0 - E_L;    // mV, rel to E_L
    constexpr static const CX::F64 LowerBound_ = -std::numeric_limits<CX::F64>::infinity();
    constexpr static const CX::F64 tau_syn_ex = 2.0; // time const. postsynaptic excitatory currents(ms)
    constexpr static const CX::F64 tau_syn_in = 2.0; // time const. postsynaptic inhibitory currents(ms)
};
template <class Tmodel = iaf_psc_alpha_default>
class iaf_psc_alpha
{
public:
    enum Channel
    {
        EXC,
        INH
    };
    struct Neuron
    {
        CX::F64vec pos;
        CX::S32 id;
        // state
        CX::F64 y0_; // Constant current
        CX::F64 dI_ex_;
        CX::F64 I_ex_;
        CX::F64 dI_in_;
        CX::F64 I_in_;

        CX::F64 y3_;
        CX::S32 r_; // refractory time remaining

        CX::F64 input_ex_;
        CX::F64 input_in_;

        CX::S32 spike;
        CX::F64 lastSpkTime;
        CX::F64 currSpkTime;
        CX::S32 spikeNum;
        CX::F64 Rsearch;
        CX::S32 randSeed;
        bool getSpike() const { return this->spike; }
        CX::F64 getCharge() const { return 0; }
        CX::F64vec getPos() const { return this->pos; }
        CX::F64 getRSearch() const { return 0; }
        void setPos(const CX::F64vec &pos) { this->pos = pos; }
        void setInputEXC(const CX::F64 input) { this->input_ex_ = input; }
        void setInputINH(const CX::F64 input) { this->input_in_ = input; }
        void addInputEXC(const CX::F64 input) { this->input_ex_ += input; }
        void addInputINH(const CX::F64 input) { this->input_in_ += input; }
        CX::F64 &getInputEXC() { return this->input_ex_; }
        CX::F64 &getInputINH() { return this->input_in_; }
        CX::F64 &getInput(Channel channel) { return channel == Channel::EXC ? this->input_ex_ : this->input_in_; }
        void clearInputEXC() { this->input_ex_ = 0; }
        void clearInputINH() { this->input_in_ = 0; }
        void init(const CX::S32 rand_seed)
        {
            this->y0_ = 0;
            this->dI_ex_ = 0;
            this->I_ex_ = 0;
            this->dI_in_ = 0;
            this->I_in_ = 0;
            this->y3_ = 5.7;
            this->r_ = 0;

            this->spike = 0;
            this->spikeNum = 0;
            this->input_ex_ = 0;
            this->input_in_ = 0;
            this->Rsearch = 1.0;
            this->currSpkTime = 0;
            this->lastSpkTime = 0;
            this->randSeed = rand_seed;
        }
        void SetInputCurrent(const CX::F64 InputCurrent) { this->y0_ = InputCurrent; }
        void SetRef()
        {
            this->r_ = Tmodel::t_ref;
            this->y3_ = Tmodel::V_reset;
        }
        void UpdateRef(const CX::F64 dt) { this->r_--; }
        void GetSpike(const CX::F64 SpkTime)
        {
            this->spike = true;
            this->lastSpkTime = this->currSpkTime;
            this->currSpkTime = SpkTime;
        }
        void NoSpike() { this->spike = false; }
        bool CheckRef() { return this->r_ == 0; }
        void setRandSeed(const CX::S32 randSeed) { this->randSeed = randSeed; }
        void recordSpike() { this->spikeNum += this->spike; }
        void writeAscii(FILE *fp) const
        {
            fprintf(fp,
                    "%u\t%lf\t%lf\t%lf\t%d\t%lf\t%lf\n",
                    this->id,
                    this->y3_,
                    this->input_ex_,
                    this->input_in_,
                    this->spikeNum,
                    this->pos.x, this->pos.y);
        }
        void readAscii(FILE *fp)
        {
            fscanf(fp,
                   "%u\t%lf\t%lf\t%lf\t%lf\t%lf\n",
                   &this->id,
                   &this->y3_,
                   &this->input_ex_,
                   &this->input_in_,
                   &this->pos.x, &this->pos.y);
        }
    };
    struct Spike // ep_j Essential Spike for scatter require
    {
        CX::F64vec pos;
        CX::F64 lastSpkTime;
        CX::F64 currSpkTime;
        CX::F64 Rsearch;
        CX::S32 id;
        CX::S32 randSeed;
        CX::F64vec getPos() const { return this->pos; }
        CX::F64 getRSearch() const { return this->Rsearch; }
        void setPos(const CX::F64vec &pos) { this->pos = pos; }
        Spike(){};
        Spike(const Neuron &rp)
            : pos(rp.pos),
              id(rp.id),
              lastSpkTime(rp.lastSpkTime),
              currSpkTime(rp.currSpkTime),
              randSeed(rp.randSeed),
              Rsearch(rp.Rsearch){};
    };
    class CalcDynamics
    {
    private:
        const CX::F64 time;
        const CX::F64 dt;
        const CX::F64 h;

        const CX::F64 EPSCInitialValue_;
        const CX::F64 IPSCInitialValue_;

        const CX::F64 P11_ex_;
        const CX::F64 P21_ex_;
        const CX::F64 P22_ex_;
        const CX::F64 P31_ex_;
        const CX::F64 P32_ex_;
        const CX::F64 P11_in_;
        const CX::F64 P21_in_;
        const CX::F64 P22_in_;
        const CX::F64 P31_in_;
        const CX::F64 P32_in_;
        const CX::F64 P30_;
        const CX::F64 P33_;
        const CX::F64 expm1_tau_m_;

    public:
        CalcDynamics(const CX::F64 _time,
                     const CX::F64 _dt)
            : time(_time),
              dt(_dt),
              h(_dt),
              EPSCInitialValue_(1.0 * std::exp(1) / Tmodel::tau_syn_ex),
              IPSCInitialValue_(1.0 * std::exp(1) / Tmodel::tau_syn_in),
              P11_ex_(std::exp(-h / Tmodel::tau_syn_ex)),
              P22_ex_(std::exp(-h / Tmodel::tau_syn_ex)), // exp_tau_syn_ex_
              P11_in_(std::exp(-h / Tmodel::tau_syn_in)),
              P22_in_(std::exp(-h / Tmodel::tau_syn_in)),
              P33_(std::exp(-h / Tmodel::tau_m)),
              expm1_tau_m_(std::expm1(-h / Tmodel::tau_m)),
              P30_(-Tmodel::tau_m / Tmodel::C_m * std::expm1(-h / Tmodel::tau_m)),
              P21_ex_(h * P11_ex_),
              P21_in_(h * P11_in_),
              P31_ex_(propagator_31(Tmodel::tau_syn_ex, Tmodel::tau_m, Tmodel::C_m, h)),
              P32_ex_(propagator_32(Tmodel::tau_syn_ex, Tmodel::tau_m, Tmodel::C_m, h)),
              P31_in_(propagator_31(Tmodel::tau_syn_in, Tmodel::tau_m, Tmodel::C_m, h)),
              P32_in_(propagator_32(Tmodel::tau_syn_in, Tmodel::tau_m, Tmodel::C_m, h)){
                  /* static bool runOnce = true;
                  if (runOnce && CX::Comm::getRank() == 0)
                  {
                      std::cout << "P11_ex_ " << P11_ex_ << std::endl;
                      std::cout << "P11_in_ " << P11_in_ << std::endl;
                      std::cout << "P22_ex_ " << P22_ex_ << std::endl;
                      std::cout << "P22_in_ " << P22_in_ << std::endl;
                      std::cout << "P33_ " << P33_ << std::endl;
                      std::cout << "expm1_tau_m_ " << expm1_tau_m_ << std::endl;
                      std::cout << "P30_ " << P30_ << std::endl;
                      std::cout << "P21_ex_ " << P21_ex_ << std::endl;
                      std::cout << "P21_in_ " << P21_in_ << std::endl;
                      std::cout << "P31_ex_ " << P31_ex_ << std::endl;
                      std::cout << "P32_ex_ " << P32_ex_ << std::endl;
                      std::cout << "P31_in_ " << P31_in_ << std::endl;
                      std::cout << "P32_in_ " << P32_in_ << std::endl;
                      std::cout << "EPSCInitialValue_ " << EPSCInitialValue_ << std::endl;
                      std::cout << "IPSCInitialValue_ " << IPSCInitialValue_ << std::endl;
                      runOnce = false;
                  } */
              };
        void operator()(CX::NeuronInstance<Neuron> &neuron)
        {
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel for
#endif
            for (CX::S32 i = 0; i < neuron.getNumberOfParticleLocal(); ++i)
            {
                if (neuron[i].r_ == 0)
                {
                    neuron[i].y3_ = P30_ *
                                        (neuron[i].y0_ + Tmodel::I_e) +
                                    P31_ex_ * neuron[i].dI_ex_ +
                                    P32_ex_ * neuron[i].I_ex_ +
                                    P31_in_ * neuron[i].dI_in_ +
                                    P32_in_ * neuron[i].I_in_ +
                                    expm1_tau_m_ * neuron[i].y3_ +
                                    neuron[i].y3_;
                    neuron[i].y3_ = neuron[i].y3_ < Tmodel::LowerBound_ ? Tmodel::LowerBound_ : neuron[i].y3_;
                }
                else
                {
                    neuron[i].r_--;
                    neuron[i].spike = 0;
                }

                neuron[i].I_ex_ = P21_ex_ * neuron[i].dI_ex_ + P22_ex_ * neuron[i].I_ex_;
                neuron[i].dI_ex_ *= P11_ex_;

                const CX::F64 weighted_spikes_ex_ = neuron[i].input_ex_;
                neuron[i].dI_ex_ += EPSCInitialValue_ * weighted_spikes_ex_;

                neuron[i].I_in_ = P21_in_ * neuron[i].dI_in_ + P22_in_ * neuron[i].I_in_;
                neuron[i].dI_in_ *= P11_in_;

                const CX::F64 weighted_spikes_in_ = neuron[i].input_in_;
                neuron[i].dI_in_ += IPSCInitialValue_ * weighted_spikes_in_;

                if (neuron[i].y3_ >= Tmodel::V_th)
                {
                    neuron[i].GetSpike(time);
                    neuron[i].y3_ = Tmodel::V_reset;
                    neuron[i].r_ = Tmodel::t_ref;
                }
                neuron[i].input_ex_ = 0;
                neuron[i].input_in_ = 0; // clear input channel
            }
        }
    };
    class TestDynamics
    {
    private:
        const CX::F64 time;
        const CX::F64 h_ms_;

        const CX::F64 psc_norm_ex_;    //!< e / tau_syn_ex
        const CX::F64 psc_norm_in_;    //!< e / tau_syn_in
        const CX::F64 expm1_tau_m_;    //!< exp(-h/tau_m) - 1
        const CX::F64 exp_tau_syn_ex_; //!< exp(-h/tau_syn_ex)
        const CX::F64 exp_tau_syn_in_; //!< exp(-h/tau_syn_in)
        const CX::F64 P30_;            //!< progagator matrix elem, 3rd row
        const CX::F64 P31_ex_;         //!< progagator matrix elem, 3rd row (ex)
        const CX::F64 P32_ex_;         //!< progagator matrix elem, 3rd row (ex)
        const CX::F64 P31_in_;         //!< progagator matrix elem, 3rd row (in)
        const CX::F64 P32_in_;         //!< progagator matrix elem, 3rd row (in)

    public:
        TestDynamics(const CX::F64 _time,
                     const CX::F64 _dt)
            : time(_time),
              h_ms_(_dt),
              psc_norm_ex_(1.0 * std::exp(1) / Tmodel::tau_syn_ex),
              psc_norm_in_(1.0 * std::exp(1) / Tmodel::tau_syn_in),
              expm1_tau_m_(std::expm1(-h_ms_ / Tmodel::tau_m)),
              exp_tau_syn_ex_(std::exp(-h_ms_ / Tmodel::tau_syn_ex)),
              exp_tau_syn_in_(std::exp(-h_ms_ / Tmodel::tau_syn_in)),
              P30_(-Tmodel::tau_m / Tmodel::C_m * expm1_tau_m_),
              P31_ex_(propagator_31(Tmodel::tau_syn_ex, Tmodel::tau_m, Tmodel::C_m, h_ms_)),
              P32_ex_(propagator_32(Tmodel::tau_syn_ex, Tmodel::tau_m, Tmodel::C_m, h_ms_)),
              P31_in_(propagator_31(Tmodel::tau_syn_in, Tmodel::tau_m, Tmodel::C_m, h_ms_)),
              P32_in_(propagator_32(Tmodel::tau_syn_in, Tmodel::tau_m, Tmodel::C_m, h_ms_)){};
        void operator()(CX::NeuronInstance<Neuron> &neuron)
        {
            for (CX::S32 i = 0; i < neuron.getNumberOfParticleLocal(); ++i)
            {    
                if (neuron[i].r_ == 0)
                {
                    neuron[i].y3_ = P30_ *
                                        (neuron[i].y0_ + Tmodel::I_e) +
                                    P31_ex_ * neuron[i].dI_ex_ +
                                    P32_ex_ * neuron[i].I_ex_ +
                                    P31_in_ * neuron[i].dI_in_ +
                                    P32_in_ * neuron[i].I_in_ +
                                    expm1_tau_m_ * neuron[i].y3_ +
                                    neuron[i].y3_;
                    neuron[i].y3_ = neuron[i].y3_ < Tmodel::LowerBound_ ? Tmodel::LowerBound_ : neuron[i].y3_;
                }
                else
                {
                    neuron[i].r_--;
                    neuron[i].spike = 0;
                }

                neuron[i].I_ex_ = exp_tau_syn_ex_ * h_ms_ * neuron[i].dI_ex_ + exp_tau_syn_ex_ * neuron[i].I_ex_;
                neuron[i].dI_ex_ = exp_tau_syn_ex_ * neuron[i].dI_ex_;

                const CX::F64 weighted_spikes_ex_ = neuron[i].input_syn_ex_;
                neuron[i].dI_ex_ += psc_norm_ex_ * weighted_spikes_ex_;

                neuron[i].I_in_ = exp_tau_syn_in_ * h_ms_ * neuron[i].dI_in_ + exp_tau_syn_in_ * neuron[i].I_in_;
                neuron[i].dI_in_ = exp_tau_syn_in_ * neuron[i].dI_in_;

                const CX::F64 weighted_spikes_in_ = neuron[i].input_syn_in_;
                neuron[i].dI_in_ += psc_norm_in_ * weighted_spikes_in_;

                if (neuron[i].y3_ >= Tmodel::V_th)
                {
                    neuron[i].GetSpike(time);
                    neuron[i].y3_ = Tmodel::V_reset;
                    neuron[i].r_ = Tmodel::t_ref;
                }

                neuron[i].input_syn_ex_ = 0;
                neuron[i].input_syn_in_ = 0; // clear input channel
            }
        }
    };
    class CheckDynamicsImpl
    {
    private:
        const CX::F64 time;
        const CX::F64 dt;
        const CX::F64 h;

        const CX::F64 EPSCInitialValue_;
        const CX::F64 IPSCInitialValue_;

        const CX::F64 P11_ex_;
        const CX::F64 P21_ex_;
        const CX::F64 P22_ex_;
        const CX::F64 P31_ex_;
        const CX::F64 P32_ex_;
        const CX::F64 P11_in_;
        const CX::F64 P21_in_;
        const CX::F64 P22_in_;
        const CX::F64 P31_in_;
        const CX::F64 P32_in_;
        const CX::F64 P30_;
        const CX::F64 P33_;
        const CX::F64 expm1_tau_m_;

        /* const CX::F64 weighted_spikes_ex_;
        const CX::F64 weighted_spikes_in_; */

    public:
        CheckDynamicsImpl(const CX::F64 _time,
                          const CX::F64 _dt)
            : time(_time),
              dt(_dt),
              h(_dt),
              EPSCInitialValue_(1.0 * std::exp(1) / Tmodel::tau_syn_ex),
              IPSCInitialValue_(1.0 * std::exp(1) / Tmodel::tau_syn_in),
              P11_ex_(std::exp(-h / Tmodel::tau_syn_ex)),
              P22_ex_(std::exp(-h / Tmodel::tau_syn_ex)),
              P11_in_(std::exp(-h / Tmodel::tau_syn_in)),
              P22_in_(std::exp(-h / Tmodel::tau_syn_in)),
              P33_(std::exp(-h / Tmodel::tau_m)),
              expm1_tau_m_(std::expm1(-h / Tmodel::tau_m)),
              P30_(-Tmodel::tau_m / Tmodel::C_m * std::expm1(-h / Tmodel::tau_m)),
              P21_ex_(h * P11_ex_),
              P21_in_(h * P11_in_),
              P31_ex_(propagator_31(Tmodel::tau_syn_ex, Tmodel::tau_m, Tmodel::C_m, h)),
              P32_ex_(propagator_32(Tmodel::tau_syn_ex, Tmodel::tau_m, Tmodel::C_m, h)),
              P31_in_(propagator_31(Tmodel::tau_syn_in, Tmodel::tau_m, Tmodel::C_m, h)),
              P32_in_(propagator_32(Tmodel::tau_syn_in, Tmodel::tau_m, Tmodel::C_m, h)){};
        void operator()(CX::NeuronInstance<Neuron> &neuron)
        {
            CX::F64 sum_input_exc = 0;
            CX::F64 sum_input_inh = 0;
            for (CX::S32 i = 0; i < neuron.getNumberOfParticleLocal(); ++i)
            {
                sum_input_exc += neuron[i].input_syn_ex_;
                sum_input_inh += neuron[i].input_syn_in_;
                neuron[i].input_syn_ex_ = 0;
                neuron[i].input_syn_in_ = 0;
            }
            sum_input_exc = CX::Comm::getSum(sum_input_exc);
            sum_input_inh = CX::Comm::getSum(sum_input_inh);
            CX::F64 neuron_sum = neuron.getNumberOfParticleGlobal();
            /*             if(sum_input_exc/neuron_sum != 9000 && sum_input_exc/neuron_sum != 2250 && sum_input_exc/neuron_sum != 0)
                        {
                            CX::Abort(-1);
                        } */
            if (CX::Comm::getRank() == 0)
            {
                std::cout << "avg_input_exc " << sum_input_exc / neuron_sum << std::endl;
                std::cout << "avg_input_inh " << sum_input_inh / neuron_sum << std::endl;
            }
        }
    };
};