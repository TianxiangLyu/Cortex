#include <cortex.hpp>
#include <limits>
#include <stdlib.h>
#include <cortex_utils.hpp>
#include <random>
namespace CX = Cortex;
struct iaf_psc_exp_default
{
    // neuron
    constexpr static const CX::F64 Tau_ = 10.0;             // Membrane time constant(ms)
    constexpr static const CX::F64 C_ = 250.0;              // Capacity of the membrane(pF)
    constexpr static const CX::F64 t_ref_ = 2.0;            // Duration of refractory period(ms)
    constexpr static const CX::F64 E_L_ = -70.0;            // Resting membrane potential(mV)
    constexpr static const CX::F64 I_e_ = 0.0;              // Reset Potential(mV)
    constexpr static const CX::F64 V_reset_ = -70.0 - E_L_; // mV, rel to E_L
    constexpr static const CX::F64 Theta_ = -55.0 - E_L_;   // mV, rel to E_L
    constexpr static const CX::F64 rho_ = 0.01;             // in 1/s
    constexpr static const CX::F64 delta_ = 0.0;            // mV, rel to E_L
    constexpr static const CX::F64 tau_ex_ = 2.0;           // time const. postsynaptic excitatory currents(ms)
    constexpr static const CX::F64 tau_in_ = 2.0;           // time const. postsynaptic inhibitory currents(ms)
};
template <class Tmodel = iaf_psc_exp_default>
class iaf_psc_exp
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
        CX::F64 Rsearch;
        CX::S32 randSeed;
        // state
        CX::F64 i_0_;      //!< Stepwise constant input current
        CX::F64 i_1_;      //!< Current input that is filtered through the excitatory synapse exponential kernel
        CX::F64 i_syn_ex_; //!< Postsynaptic current for excitatory inputs (includes contribution from current input on
                           //!< receptor type 1)
        CX::F64 i_syn_in_; //!< Postsynaptic current for inhibitory inputs
        CX::F64 V_m_;      //!< Membrane potential

        CX::F64 r_ref_; //!< Absolute refractory counter (no membrane potential propagation)

        CX::F64 input_syn_ex_;
        CX::F64 input_syn_in_;

        CX::F64 input_i0_;
        CX::F64 input_i1_;

        // spike record
        CX::S32 spike;
        CX::F64 lastSpkTime;
        CX::F64 currSpkTime;
        CX::S32 spikeNum;
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
        CX::F64 &getInput(Channel channel) { return channel == Channel::EXC ? this->input_syn_ex_ : this->input_syn_in_; }
        void clearInputEXC() { this->input_ex_ = 0; }
        void clearInputINH() { this->input_in_ = 0; }
        void init(const CX::S32 rand_seed)
        {
            this->Rsearch = 1.0;
            this->randSeed = rand_seed;
            this->i_0_ = 0;
            this->i_1_ = 0;
            this->i_syn_ex_ = 0;
            this->i_syn_in_ = 0;
            this->input_syn_ex_ = 0;
            this->input_syn_in_ = 0;
            this->input_i0_ = 0;
            this->input_i1_ = 0;
            this->V_m_ = 0;
            this->r_ref_ = 0;

            this->spike = 0;
            this->spikeNum = 0;
            this->currSpkTime = 0;
            this->lastSpkTime = 0;
        }
        void SetRef()
        {
            this->r_ref_ = Tmodel::t_ref_;
            this->V_m_ = Tmodel::V_reset_;
        }
        void UpdateRef(const CX::F64 dt) { this->r_ref_--; }
        void GetSpike(const CX::F64 SpkTime)
        {
            this->spike = true;
            this->lastSpkTime = this->currSpkTime;
            this->currSpkTime = SpkTime;
        }
        void NoSpike() { this->spike = false; }
        bool CheckRef() { return this->r_ref_ == 0; }
        void setRandSeed(const CX::S32 randSeed) { this->randSeed = randSeed; }
        void recordSpike() { this->spikeNum += this->spike; }
        void writeAscii(FILE *fp) const
        {
            fprintf(fp,
                    "%u\t%lf\t%lf\t%lf\t%d\t%lf\t%lf\n",
                    this->id,
                    this->V_m_,
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

        const CX::F64 P11ex_;
        const CX::F64 P11in_;
        const CX::F64 P20_;
        const CX::F64 P21ex_;
        const CX::F64 P21in_;
        const CX::F64 P22_;

    public:
        CalcDynamics(const CX::F64 _time,
                     const CX::F64 _dt)
            : time(_time),
              dt(_dt),
              h(_dt),
              P11ex_(std::exp(-h / Tmodel::tau_ex_)),
              P11in_(std::exp(-h / Tmodel::tau_in_)),
              P21ex_(propagator_32(Tmodel::tau_ex_, Tmodel::Tau_, Tmodel::C_, h)),
              P21in_(propagator_32(Tmodel::tau_in_, Tmodel::Tau_, Tmodel::C_, h)),
              P22_(std::exp(-h / Tmodel::Tau_)),
              P20_(Tmodel::Tau_ / Tmodel::C_ * (1.0 - std::exp(-h / Tmodel::Tau_)))
        {
            static bool runOnce = true;
            if (runOnce && CX::Comm::getRank() == 0)
            {
                std::cout << "P11ex_ " << P11ex_ << std::endl;
                std::cout << "P11in_ " << P11in_ << std::endl;
                std::cout << "P21ex_ " << P21ex_ << std::endl;
                std::cout << "P21in_ " << P21in_ << std::endl;
                std::cout << "P22_ " << P22_ << std::endl;
                std::cout << "P20_ " << P20_ << std::endl;
                runOnce = false;
            }
        };
        CX::F64 phi(const CX::F64 V_m)
        {
            assert(Tmodel::delta_ > 0.);
            return Tmodel::rho_ * std::exp(1. / Tmodel::delta_ * (V_m - Tmodel::Theta_));
        }
        void operator()(CX::NeuronInstance<Neuron> &neuron)
        {
            std::random_device rd;
            std::default_random_engine eng(rd());
            std::uniform_real_distribution<CX::F64> d(0.0, 1.0);
            /* #ifdef CORTEX_THREAD_PARALLEL
            #pragma omp parallel for
            #endif */
            for (CX::S32 i = 0; i < neuron.getNumberOfParticleLocal(); ++i)
            {
                if (neuron[i].r_ref_ <= 0)
                {
                    neuron[i].V_m_ = neuron[i].V_m_ * P22_ + neuron[i].i_syn_ex_ * P21ex_ + neuron[i].i_syn_in_ * P21in_ + (Tmodel::I_e_ + neuron[i].i_0_) * P20_;
                }
                else
                {
                    neuron[i].r_ref_ -= h;
                    neuron[i].spike = 0;
                }

                neuron[i].i_syn_ex_ *= P11ex_;
                neuron[i].i_syn_in_ *= P11in_;

                neuron[i].i_syn_ex_ += (1. - P11ex_) * neuron[i].i_1_;

                const CX::F64 weighted_spikes_ex_ = neuron[i].input_syn_ex_;
                const CX::F64 weighted_spikes_in_ = neuron[i].input_syn_in_;

                neuron[i].i_syn_ex_ += weighted_spikes_ex_;
                neuron[i].i_syn_in_ += weighted_spikes_in_;
                if ((Tmodel::delta_ < 1e-10 && neuron[i].V_m_ >= Tmodel::Theta_) 
                || (Tmodel::delta_ > 1e-10 && d(eng) < phi(neuron[i].V_m_) * h * 1e-3))
                {
                    neuron[i].GetSpike(time);
                    neuron[i].V_m_ = Tmodel::V_reset_;
                    neuron[i].r_ref_ = Tmodel::t_ref_;
                }
                neuron[i].input_syn_ex_ = 0;
                neuron[i].input_syn_in_ = 0; // clear input channel
            }
        }
    };
};