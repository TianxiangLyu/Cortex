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
        aligned_vector<typename Tsyn::Synapse> epj_act_;
        aligned_vector<typename Tsyn::LinkInfo> epj_link_;
        aligned_vector<typename Tsyn::Post> epi_org_; // add Input Channel inside
        DelayQueue<typename Tsrc::Spike> &queue_;
        const std::vector<typename Tsrc::Spike> &spk_tot_;
        NeuronInstance<typename Tdst::Neuron> &dst_neuron_;

    public:
        template <class Tneu_src, class Tspk_src,
                  class Tneu_dst, class Tspk_dst,
                  class TChannel>
        Connection(LayerInfo<Tneu_src, Tspk_src> &src,
                   LayerInfo<Tneu_dst, Tspk_dst> &dst,
                   TChannel Channel,
                   F64 delay = Tsyn::delay)
            : conn_id_(num_conn_glb_++),
              delay_(delay),
              queue_(src.queue_),
              spk_tot_(src.spk_tot_),
              dst_neuron_(dst.neuron_)
        {
            const S32 n_epi = dst.getNumLocal();
            src.addConn(conn_id_, delay, dst.getDinfo());
            if (n_epi > 0)
            {
                epi_org_.reserve(n_epi);
                for (S32 i = 0; i < n_epi; i++)
                    epi_org_.push_back(typename Tsyn::Post(dst[i], dst[i].getInput(Channel)));
                const S32 n_epj = src.getNumGlobal();
                epj_link_.resize(n_epj);
            }
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
            pfunc_ep_ep(epi_org_.data(), epi_org_.size(),
                        epj_act_.data(), epj_act_.size());
        }
        void PostSpk(const F64 time, const F64 dt)
        {
            if (epi_org_.size() == 0)
                return;
        }
        void SetIndegreeMultapsesOMP(const S32 indegree)
        {
            if (epi_org_.size() == 0)
                return;
            const S32 n_threads = Comm::getNumberOfThread();
            const S32 n_epi = epi_org_.size();
            const S32 n_epj = epj_link_.size();
            std::vector<S32> link_num(n_epj, 0);
            std::vector<S32> ptr(n_epj, 0);
#ifdef CORTEX_THREAD_PARALLEL
            std::vector<omp_lock_t> lock(n_epj);
            for (S32 j = 0; j < n_epj; j++)
                omp_init_lock(&(lock[j]));
#endif
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel
#endif
            {
                const S32 ith = Comm::getThreadNum();
                std::vector<S32> link_num_ith(n_epj, 0);
                std::uniform_int_distribution<S32> distr(0, n_epj - 1);
                std::default_random_engine test(epi_org_[0].randSeed);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
                for (S32 i = 0; i < n_epi; i++)
                {
                    std::default_random_engine eng(epi_org_[i].randSeed);
                    for (S32 j = 0; j < indegree; j++)
                    {
                        S32 adr = distr(eng);
                        while (adr == epi_org_[i].id)
                            adr = distr(eng);
                        link_num_ith[adr]++;
                    }
                }
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp critical
#endif
                for (S32 j = 0; j < n_epj; j++)
                    link_num[j] += link_num_ith[j];
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp barrier // important
#endif
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
                for (S32 j = 0; j < n_epj; j++)
                    epj_link_[j].init(link_num[j]);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
                for (S32 i = 0; i < n_epi; i++)
                {
                    std::default_random_engine eng(epi_org_[i].randSeed);
                    for (S32 j = 0; j < indegree; j++)
                    {
                        S32 adr = distr(eng);
                        while (adr == epi_org_[i].id)
                            adr = distr(eng);
#ifdef CORTEX_THREAD_PARALLEL
                        omp_set_lock(&(lock[adr]));
#endif
                        epj_link_[adr].setLink(ptr[adr], i);
                        ptr[adr]++;
#ifdef CORTEX_THREAD_PARALLEL
                        omp_unset_lock(&(lock[adr]));
#endif
                    }
                }
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
                for (S32 j = 0; j < n_epj; j++)
                    std::sort(epj_link_[j].info, epj_link_[j].info + epj_link_[j].n_link,
                              [](const typename Tsyn::Link &l, const typename Tsyn::Link &r)
                                  -> bool
                              { return l.target < r.target; });
            } // end omp
            /* for (S32 j = 0; j < n_epj; j++)
                omp_destroy_lock(&(lock[j]));
            for (S32 j = 0; j < n_epj; j++)
                assert(ptr[j] == link_num[j]);
            std::vector<S32> link_num_check(n_epj);
            std::vector<S32> ptr_check(n_epj, 0);
            std::vector<typename Tsyn::LinkInfo> epj_link_check(n_epj);
            std::uniform_int_distribution<S32> distr(0, n_epj - 1);
            for (S32 i = 0; i < n_epi; i++)
            {
                std::default_random_engine eng(epi_org_[i].randSeed);
                for (S32 j = 0; j < indegree; j++)
                {
                    S32 adr = distr(eng);
                    while (adr == epi_org_[i].id)
                        adr = distr(eng);
                    assert(adr != epi_org_[i].id);
                    link_num_check[adr]++;
                }
            }
            for (S32 j = 0; j < n_epj; j++)
                epj_link_check[j].init(link_num[j]);
            for (S32 i = 0; i < n_epi; i++)
            {
                std::default_random_engine eng(epi_org_[i].randSeed);
                for (S32 j = 0; j < indegree; j++)
                {
                    S32 adr = distr(eng);
                    while (adr == epi_org_[i].id)
                        adr = distr(eng);
                    assert(adr != epi_org_[i].id);
                    if (ptr_check[adr] >= epj_link_check[adr].n_link)
                    {
                        std::cout << adr << " " << ptr_check[adr] << " " << epj_link_check[adr].n_link << std::endl;
                        Abort(-1);
                    }
                    epj_link_check[adr].setLink(ptr_check[adr], i);
                    ptr_check[adr]++;
                }
            }
            for (S32 j = 0; j < n_epj; j++)
            {
                if (link_num[j] != link_num_check[j])
                {
                    std::cout << "j " << j << " link_num[j] " << link_num[j] << " link_num_check[j] " << link_num_check[j] << std::endl;
                    Abort(-1);
                }
                if (ptr[j] != ptr_check[j])
                {
                    std::cout << "j " << j << " ptr[j] " << ptr[j] << " ptr_check[j] " << ptr_check[j] << std::endl;
                    Abort(-1);
                }
                if (epj_link_[j].n_link != epj_link_check[j].n_link)
                {
                    std::cout << "j " << j << " epj_link[j].n_link " << epj_link_[j].n_link << " epj_link_check[j].n_link " << epj_link_check[j].n_link << std::endl;
                    Abort(-1);
                }
            } */
        }
        void SetIndegreeMultapsesAutapsesOMP(const S32 indegree)
        {
            if (epi_org_.size() == 0)
                return;
            const S32 n_threads = Comm::getNumberOfThread();
            const S32 n_epi = epi_org_.size();
            const S32 n_epj = epj_link_.size();
            std::vector<S32> link_num(n_epj, 0);
            std::vector<S32> ptr(n_epj, 0);
#ifdef CORTEX_THREAD_PARALLEL
            std::vector<omp_lock_t> lock(n_epj);
            for (S32 j = 0; j < n_epj; j++)
                omp_init_lock(&(lock[j]));
#endif
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel
#endif
            {
                const S32 ith = Comm::getThreadNum();
                std::vector<S32> link_num_ith(n_epj, 0);
                std::uniform_int_distribution<S32> distr(0, n_epj - 1);
                std::default_random_engine test(epi_org_[0].randSeed);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
                for (S32 i = 0; i < n_epi; i++)
                {
                    std::default_random_engine eng(epi_org_[i].randSeed);
                    for (S32 j = 0; j < indegree; j++)
                    {
                        S32 adr = distr(eng);
                        link_num_ith[adr]++;
                    }
                }
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp critical
#endif
                for (S32 j = 0; j < n_epj; j++)
                    link_num[j] += link_num_ith[j];
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp barrier // important
#endif
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
                for (S32 j = 0; j < n_epj; j++)
                    epj_link_[j].init(link_num[j]);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
                for (S32 i = 0; i < n_epi; i++)
                {
                    std::default_random_engine eng(epi_org_[i].randSeed);
                    for (S32 j = 0; j < indegree; j++)
                    {
                        S32 adr = distr(eng);
#ifdef CORTEX_THREAD_PARALLEL
                        omp_set_lock(&(lock[adr]));
#endif
                        epj_link_[adr].setLink(ptr[adr], i);
                        ptr[adr]++;
#ifdef CORTEX_THREAD_PARALLEL
                        omp_unset_lock(&(lock[adr]));
#endif
                    }
                }
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
                for (S32 j = 0; j < n_epj; j++)
                    std::sort(epj_link_[j].info, epj_link_[j].info + epj_link_[j].n_link,
                              [](const typename Tsyn::Link &l, const typename Tsyn::Link &r)
                                  -> bool
                              { return l.target < r.target; });
            } // end omp
            /* for (S32 j = 0; j < n_epj; j++)
                omp_destroy_lock(&(lock[j]));
            for (S32 j = 0; j < n_epj; j++)
                assert(ptr[j] == link_num[j]);
            std::vector<S32> link_num_check(n_epj);
            std::vector<S32> ptr_check(n_epj, 0);
            std::vector<typename Tsyn::LinkInfo> epj_link_check(n_epj);
            std::uniform_int_distribution<S32> distr(0, n_epj - 1);
            for (S32 i = 0; i < n_epi; i++)
            {
                std::default_random_engine eng(epi_org_[i].randSeed);
                for (S32 j = 0; j < indegree; j++)
                {
                    S32 adr = distr(eng);
                    while (adr == epi_org_[i].id)
                        adr = distr(eng);
                    assert(adr != epi_org_[i].id);
                    link_num_check[adr]++;
                }
            }
            for (S32 j = 0; j < n_epj; j++)
                epj_link_check[j].init(link_num[j]);
            for (S32 i = 0; i < n_epi; i++)
            {
                std::default_random_engine eng(epi_org_[i].randSeed);
                for (S32 j = 0; j < indegree; j++)
                {
                    S32 adr = distr(eng);
                    while (adr == epi_org_[i].id)
                        adr = distr(eng);
                    assert(adr != epi_org_[i].id);
                    if (ptr_check[adr] >= epj_link_check[adr].n_link)
                    {
                        std::cout << adr << " " << ptr_check[adr] << " " << epj_link_check[adr].n_link << std::endl;
                        Abort(-1);
                    }
                    epj_link_check[adr].setLink(ptr_check[adr], i);
                    ptr_check[adr]++;
                }
            }
            for (S32 j = 0; j < n_epj; j++)
            {
                if (link_num[j] != link_num_check[j])
                {
                    std::cout << "j " << j << " link_num[j] " << link_num[j] << " link_num_check[j] " << link_num_check[j] << std::endl;
                    Abort(-1);
                }
                if (ptr[j] != ptr_check[j])
                {
                    std::cout << "j " << j << " ptr[j] " << ptr[j] << " ptr_check[j] " << ptr_check[j] << std::endl;
                    Abort(-1);
                }
                if (epj_link_[j].n_link != epj_link_check[j].n_link)
                {
                    std::cout << "j " << j << " epj_link[j].n_link " << epj_link_[j].n_link << " epj_link_check[j].n_link " << epj_link_check[j].n_link << std::endl;
                    Abort(-1);
                }
            } */
        }
        void SetWeightAll(const CX::F64 weight)
        {
            if (epi_org_.size() == 0)
                return;
            const S32 n_epj = epj_link_.size();
            // assert(n_epj == spk_tot_.size());
            for (S32 j = 0; j < n_epj; j++)
                for (S32 i = 0; i < epj_link_[j].n_link; i++)
                    epj_link_[j].setWeight(i, weight);
        }
        void SetWeightAllOMP(const CX::F64 weight)
        {
            if (epi_org_.size() == 0)
                return;
            const S32 n_epj = epj_link_.size();
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel for
#endif
            for (S32 j = 0; j < n_epj; j++)
                for (S32 i = 0; i < epj_link_[j].n_link; i++)
                    epj_link_[j].setWeight(i, weight);
        }
        void CountLink()
        {
            S64 n_link = 0;
            S64 n_null = 0;
            for (S32 i = 0; i < epj_link_.size(); i++)
            {
                n_link += epj_link_[i].n_link;
                if (epj_link_[i].n_link == 0)
                    n_null++;
            }
            n_link = Comm::getSum(n_link);
            n_null = Comm::getSum(n_null);
            if (Comm::getRank() == 0)
            {
                std::cout << "num of total link " << n_link << std::endl;
                std::cout << "num of null link " << n_null << std::endl;
            }
        }
    };
}