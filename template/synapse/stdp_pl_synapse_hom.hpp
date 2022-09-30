#include <cortex.hpp>
#include <atomic>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <unordered_set>
namespace CX = Cortex;
template <class params> // Add InputChannel?
class stdp_pl_synapse_hom
{
public:
    struct histentry
    {
        histentry(CX::F64 _t, CX::F64 _Kminus, CX::F64 _Kminus_triplet, CX::S32 _access_counter)
            : SpkTime(_t),
              Kminus(_Kminus),
              Kminus_triplet(_Kminus_triplet),
              access_counter(_access_counter){};
        CX::F64 SpkTime;        //!< point in time when spike occurred (in ms)
        CX::F64 Kminus;         //!< value of Kminus at that time
        CX::F64 Kminus_triplet; //!< value of triplet STDP Kminus at that time
        CX::S32 access_counter; //!< access counter to enable removal of the entry, once all neurons read it
    };
    struct spike_interval
    {
        typename CX::aligned_deque<histentry>::iterator start;
        typename CX::aligned_deque<histentry>::iterator end;
        const spike_interval &operator=(const spike_interval s)
        {
            this->start = s.start;
            this->end = s.end;
            return (*this);
        }
    };
    constexpr static const CX::F64 delay = params::delay;
    constexpr static const CX::F64 alpha = params::alpha;
    constexpr static const CX::F64 lambda = params::lambda;
    constexpr static const CX::F64 mu = params::mu;
    constexpr static const CX::F64 tau_plus = params::tau_plus;
    constexpr static const CX::F64 tau_plus_inv = 1.0 / params::tau_plus;
    constexpr static const CX::F64 offset = 0;
    constexpr static const CX::F64 stdp_eps = 1e-6;
    constexpr static const CX::F64 tau_minus = params::tau_minus;
    constexpr static const CX::F64 tau_minus_inv = 1.0 / params::tau_minus;
    constexpr static const CX::F64 tau_minus_triplet = 110.0;
    constexpr static const CX::F64 tau_minus_triplet_inv = 1.0 / tau_minus_triplet;
    struct Link
    {
        CX::S32 target;
        CX::F32 weight;
        CX::F32 Kplus;
        Link(){};
        Link(const CX::S32 _target, const CX::F64 _weight, const CX::F64 _Kplus)
            : target(_target),
              weight(_weight),
              Kplus(_Kplus){};
        Link(const CX::S32 _target, const Link &_link)
            : target(_target),
              weight(_link.weight),
              Kplus(_link.Kplus){};
    };
    struct LinkInfo
    {
        CX::S32 n_link;
        Link *info;
        void init(const CX::S32 num)
        {
            this->n_link = num;
            info = new Link[num];
        }
        void setLink(const CX::S32 id, const CX::S32 target) { this->info[id].target = target; }
        void setWeight(const CX::S32 id, const CX::F64 value) { this->info[id].weight = value; }
        ~LinkInfo()
        {
            delete[] info;
        }
    };
    struct Post
    {
        const CX::S32 id;
        const CX::F64vec pos;
        const CX::S32 randSeed;
        const CX::F64 Rsearch;
        // for STDP
        CX::F64 lastSpkTime = -1.0;
        CX::S32 n_link = 0;   // number of incoming connections
        CX::F64 Kminus = 0.0; // the current time-dependent weighting of the STDP update rule for depression
        CX::F64 Kminus_triplet = 0.0;
        CX::F64 trace = 0;
        CX::F64 &input;
        CX::aligned_deque<histentry> history;
        template <class Tfp>
        Post(Tfp &fp, CX::F64 &_input)
            : id(fp.id),
              pos(fp.pos),
              randSeed(fp.randSeed),
              Rsearch(fp.Rsearch),
              lastSpkTime(-1.0),
              n_link(0),
              Kminus(0.0),
              Kminus_triplet(0.0),
              trace(0.0),
              input(_input){};
        /*         template <class Tep>
                void setFromEP(const Tep &ep)
                {
                    this->id = ep.id;
                    this->pos = ep.pos;
                    this->randSeed = ep.randSeed;
                    this->Rsearch = ep.Rsearch;
                } */
        void clearHistory()
        {
            this->history.clear();
        }
        inline void updateSpk(const CX::F64 time, const CX::F64 max_delay = delay) // nest set_spiketime
        {
            const CX::F64 currSpkTime = time;
            if (this->n_link)
            {
                while (this->history.size() > 1)
                {
                    const CX::F64 next_t_sp = this->history[1].SpkTime;
                    if (this->history.front().access_counter >= this->n_link && currSpkTime - next_t_sp > max_delay + stdp_eps)
                        history.pop_front();
                    else
                        break;
                }
                // update spiking history
                this->Kminus = this->Kminus * std::exp((this->lastSpkTime - currSpkTime) * tau_minus_inv) + 1.0;
                this->Kminus_triplet = this->Kminus_triplet * std::exp((this->lastSpkTime - currSpkTime) * tau_minus_triplet_inv) + 1.0;
                this->lastSpkTime = currSpkTime;
                this->history.push_back(histentry(currSpkTime, this->Kminus, this->Kminus_triplet, 0));
            }
            else
                this->lastSpkTime = currSpkTime;
        }
        inline spike_interval getHistory(const CX::F64 t1, const CX::F64 t2)
        {
            spike_interval ins;
            ins.end = this->history.end();
            if (this->history.empty())
            {
                ins.start = ins.end;
                return ins;
            }
            typename CX::aligned_deque<histentry>::reverse_iterator it = history.rbegin();
            const CX::F64 t2_lim = t2 + 1e-6; // stdp_eps
            const CX::F64 t1_lim = t1 + 1e-6; // ganrantee that the spike interval is (t1, t2]
            while (it != history.rend() && it->SpkTime >= t2_lim)
                it++;
            ins.end = it.base();
            while (it != history.rend() && it->SpkTime >= t1_lim)
            {
                it->access_counter++;
                it++;
            }
            ins.start = it.base();
            return ins;
        }
        inline CX::F64 getKvalue(const CX::F64 t)
        {
            if (this->history.empty())
            {
                this->trace = 0;
                return this->trace;
            }
            // search for the latest post spike in the history buffer that came strictly
            // before `t`
            CX::S32 it = this->history.size() - 1;
            while (it >= 0)
            {
                if (t - this->history[it].SpkTime > stdp_eps)
                {
                    this->trace = this->history[it].Kminus * std::exp((this->history[it].SpkTime - t) * tau_minus_inv);
                    return this->trace;
                }
                it--;
            }
            // this case occurs when the trace was requested at a time precisely at or
            // before the first spike in the history
            this->trace = 0;
            return this->trace;
        }
    }; // position is not require
    struct Synapse
    {
        const CX::S32 id;
        const CX::F64 currSpkTime;
        const CX::F64 lastSpkTime;
        LinkInfo &link;
        template <class Tep>
        Synapse(Tep ep, LinkInfo &_link)
            : id(ep.id),
              currSpkTime(ep.currSpkTime),
              lastSpkTime(ep.lastSpkTime),
              link(_link){};
    };
    class ClearSpikeHistory // and check
    {
    public:
        void operator()(Post *const ep_i, const CX::S32 Nip,
                        Synapse *const ep_j, const CX::S32 Njp)
        {
            for (CX::S32 i = 0; i < Nip; i++)
                ep_i[i].clearHistory();
        }
    };
    class CalcInteraction
    {
    private:
        const CX::F64 time;
        inline CX::F64 facilitate(CX::F64 w, CX::F64 kplus)
        {
            return w + (params::lambda * std::pow(w, params::mu) * kplus);
        }
        inline CX::F64 depress(CX::F64 w, CX::F64 kminus)
        {
            CX::F64 new_w = w - (params::lambda * params::alpha * w * kminus);
            return new_w > 0.0 ? new_w : 0.0;
        }

    public:
        CalcInteraction(const CX::F64 _time)
            : time(_time){};
        void operator()(Post *const ep_i, const CX::S32 Nip,
                        Synapse *const ep_j, const CX::S32 Njp)
        {

            for (CX::S32 j = 0; j < Njp; j++)
            {
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel for
#endif
                for (CX::S32 i = 0; i < ep_j[j].link.n_link; i++)
                {
                    const CX::S32 adr = ep_j[j].link.info[i].target;
                    const CX::F64 drtc_delay = delay;
                    const CX::F64 currSpkTime = ep_j[j].currSpkTime;
                    const CX::F64 lastSpkTime = ep_j[j].lastSpkTime;
                    const stdp_pl_synapse_hom::spike_interval spk_his = ep_i[adr].getHistory(lastSpkTime - drtc_delay, currSpkTime - drtc_delay);
                    for (typename CX::aligned_deque<histentry>::iterator it = spk_his.start; it != spk_his.end; it++)
                    {
                        const CX::F64 minus_dt = lastSpkTime - (it->SpkTime + drtc_delay);
                        // assert(minus_dt < 0 - stdp_eps);
                        ep_j[j].link.info[i].weight = facilitate(ep_j[j].link.info[i].weight, ep_j[j].link.info[i].Kplus * std::exp(minus_dt * tau_plus_inv));
                    }
                    ep_j[j].link.info[i].weight = depress(ep_j[j].link.info[i].weight, ep_i[adr].getKvalue(time - drtc_delay));
                    ep_j[j].link.info[i].Kplus = ep_j[j].link.info[i].Kplus * std::exp((lastSpkTime - time) * tau_plus_inv) + 1.0;
                    ep_i[adr].input += ep_j[j].link.info[i].weight;
                }
            }
        }
    };
    class TestInteraction
    {
    private:
        const CX::F64 time;
        inline CX::F64 facilitate(CX::F64 w, CX::F64 kplus)
        {
            return w + (params::lambda * std::pow(w, params::mu) * kplus);
        }
        inline CX::F64 depress(CX::F64 w, CX::F64 kminus)
        {
            CX::F64 new_w = w - (params::lambda * params::alpha * w * kminus);
            return new_w > 0.0 ? new_w : 0.0;
        }

    public:
        TestInteraction(const CX::F64 _time)
            : time(_time){};
        void operator()(Post *const ep_i, const CX::S32 Nip,
                        Synapse *const ep_j, const CX::S32 Njp)
        {
            std::vector<CX::S32> epi_adr;
            std::unordered_set<CX::S32> epi_set;
            std::unordered_multimap<CX::S32, Link> link_map;
            for (CX::S32 j = 0; j < Njp; j++)
                for (CX::S32 i = 0; i < ep_j[j].link.n_link; i++)
                {
                    link_map.insert(std::pair<CX::S32, Link>(ep_j[j].link.info[i].target, Link(j, ep_j[j].link.info[i])));
                    epi_set.insert(ep_j[j].link.info[i].target);
                }
            epi_adr.reserve(epi_set.size());
            for (auto it = epi_set.begin(); it != epi_set.end(); it++)
                epi_adr.push_back(*it);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel for
#endif
            for (CX::S32 i = 0; i < epi_adr.size(); i++)
            {
                auto range = link_map.equal_range(epi_adr[i]);
                for (auto it = range.first; it != range.second; it++)
                {
                    it->second.weight = 0;
                }
            }
            for (CX::S32 j = 0; j < Njp; j++)
            {
                for (CX::S32 i = 0; i < ep_j[j].link.n_link; i++)
                {
                    const CX::S32 adr = ep_j[j].link.info[i].target;
                    const CX::F64 drtc_delay = delay;
                    const CX::F64 currSpkTime = ep_j[j].currSpkTime;
                    const CX::F64 lastSpkTime = ep_j[j].lastSpkTime;
                    const stdp_pl_synapse_hom::spike_interval spk_his = ep_i[adr].getHistory(lastSpkTime - drtc_delay, currSpkTime - drtc_delay);
                    for (auto it = spk_his.start; it != spk_his.end; it++)
                    {
                        const CX::F64 minus_dt = lastSpkTime - (it->SpkTime + drtc_delay);
                        // assert(minus_dt < 0 - stdp_eps);
                        ep_j[j].link.info[i].weight = facilitate(ep_j[j].link.info[i].weight, ep_j[j].link.info[i].Kplus * std::exp(minus_dt * tau_plus_inv));
                    }
                    ep_j[j].link.info[i].weight = depress(ep_j[j].link.info[i].weight, ep_i[adr].getKvalue(time - drtc_delay));
                    ep_j[j].link.info[i].Kplus = ep_j[j].link.info[i].Kplus * std::exp((lastSpkTime - time) * tau_plus_inv) + 1.0;
                    ep_i[adr].input += ep_j[j].link.info[i].weight;
                }
            }
        }
    };
};