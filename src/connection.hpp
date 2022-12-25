#include <cortex.hpp>
#include <set>
#include <random>
#include <map>
#include <connection_utils.hpp>
namespace Cortex
{
    static S32 num_conn_glb_ = 0;
    template <class Tsrc, class Tsyn, class Tdst>
    class Connection
    {
    private:
        const S32 conn_id_;
        const F64 delay_;
        std::unordered_map<S32, S32> map_id_to_link_;
        std::vector<typename Tsyn::Synapse> epj_act_;
        std::vector<typename Tsyn::LinkInfo> epj_link_;
        std::vector<typename Tsyn::Post> epi_org_; // add Input Channel inside
        DelayQueue<typename Tsrc::Spike> &queue_;
        const std::vector<typename Tsrc::Spike> &spk_tot_;
        NeuronInstance<typename Tdst::Neuron> &dst_neuron_;

    public:
        template <class Tneu_src, class Tspk_src,
                  class Tneu_dst, class Tspk_dst,
                  class TChannel, class Tconn_set, class Tweight_set>
        Connection(PopulationInfo<Tneu_src, Tspk_src> &src,
                   PopulationInfo<Tneu_dst, Tspk_dst> &dst,
                   TChannel Channel, Tconn_set conn_set, Tweight_set weight_set)
            : conn_id_(num_conn_glb_++),
              delay_(Tsyn::delay),
              queue_(src.queue_),
              spk_tot_(src.spk_tot_),
              dst_neuron_(dst.neuron_)
        {
            const S32 n_epi = dst.getNumLocal();
            src.addConn(conn_id_, delay_, dst.getDinfo());
            src.SpkAllGather();
            if (n_epi > 0)
            {
                epi_org_.reserve(n_epi);
                for (S32 i = 0; i < n_epi; i++)
                    epi_org_.push_back(typename Tsyn::Post(dst[i], dst[i].getInput(Channel)));
                const S32 n_epj = src.getNumGlobal();
                epj_link_.resize(n_epj);
                conn_set(epj_link_, epi_org_, spk_tot_);
                weight_set(epj_link_, epi_org_, spk_tot_);
            }
            src.freeSpkAll();
            // std::cout<<"Rank "<<Comm::getRank()<<" epi_org_.size() "<<epi_org_.size()<<std::endl;
        };
        void setEPJAct(const F64 time)
        {
            if (epi_org_.size() == 0)
                return;
            typename std::deque<DelaySlot<typename Tsrc::Spike>>::reverse_iterator slot = queue_.rfind(time, delay_);
            epj_act_.clear();
            if (slot != queue_.rend())
            {
                assert(slot->sync); // synchronized
                const S32 n_spk = slot->epj_recv_.size();
                epj_act_.reserve(n_spk);
                for (S32 j = 0; j < n_spk; j++)
                    epj_act_.push_back(typename Tsyn::Synapse(slot->epj_recv_[j], epj_link_[slot->epj_recv_[j].id]));
            }
            /* if(Comm::getRank() == 0)
            {
                std::cout<<std::endl;
                std::cout<<"epj_act_.size() "<<epj_act_.size()<<std::endl;
                std::cout<<std::endl;
            } */
        }
        typename Tsyn::Post &operator[](const S32 id) { return epi_org_[id]; }
        template <class Tfunc_ep_ep>
        void PreAct(const F64 time, Tfunc_ep_ep pfunc_ep_ep)
        {
            if (epi_org_.size() == 0)
                return;
            setEPJAct(time);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel for
#endif
            for (S32 i = 0; i < dst_neuron_.getNumberOfParticleLocal(); i++)
                if (dst_neuron_[i].spike)
                    epi_org_[i].updateSpk(time);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel
#endif
            {
                const CX::S32 ith = CX::Comm::getThreadNum();
                const CX::S32 nth = CX::Comm::getNumThreads();
                const CX::S32 begin_i = (epi_org_.size() / nth + 1) * ith;
                const CX::S32 end_i = (epi_org_.size() / nth + 1) * (ith + 1);
                pfunc_ep_ep(epi_org_.data(), begin_i, end_i,
                            epj_act_.data(), epj_act_.size());
            }
        }
        void PostSpk(const F64 time, const F64 dt)
        {
            if (epi_org_.size() == 0)
                return;
        }
    };
}