#include <cortex.hpp>
namespace CX = Cortex;
enum Multapses
{
    Multapses_YES,
    Multapses_NO,
};
enum Autapses
{
    Autapses_YES,
    Autapses_NO,
};
static CX::S32 global_rand_seed = 1;
class FixedTotalNumber
{
private:
    const CX::S32 rand_seed; // This var should be the same in all proc to generate the same random sequence
    const CX::S32 conn_num;

public:
    FixedTotalNumber(const CX::S32 _conn_num)
        : rand_seed(global_rand_seed++),
          conn_num(_conn_num){};
    FixedTotalNumber(const CX::S32 _conn_num, const CX::S32 _rand_seed)
        : rand_seed(_rand_seed),
          conn_num(_conn_num){};
    template <class TLinkInfo, class TPost, class TSpike>
    inline void operator()(std::vector<TLinkInfo> &epj_link,
                           std::vector<TPost> &epi_org,
                           const std::vector<TSpike> &spk_tot)
    {
        if (epi_org.size() == 0)
            return;
        const CX::S32 n_threads = CX::Comm::getNumberOfThread();
        const CX::S32 n_epi = epi_org.size();
        const CX::S32 n_epj = epj_link.size();
        std::vector<CX::S32> epi(conn_num);
        std::vector<CX::S32> epj(conn_num);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel
#endif
        {
            const CX::S32 ith = CX::Comm::getThreadNum();
            std::vector<CX::S32> ptr;
            std::default_random_engine eng(rand_seed); // using the same rand seed
            std::uniform_int_distribution<CX::S32> epi_distr(0, n_epi - 1);
            std::uniform_int_distribution<CX::S32> epj_distr(0, n_epj - 1);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
            for (CX::S32 k = 0; k < conn_num; k++)
            {
                epi[k] = epi_distr(eng);
                epj[k] = epj_distr(eng);
            }
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
            for (CX::S32 j = 0; j < n_epj; j++)
            {
                ptr.clear();
                for (CX::S32 k = 0; k < conn_num; k++)
                    if (epj[k] == j)
                        for (CX::S32 i = 0; i < n_epi; i++)
                            if (epi_org[i].id == epi[k])
                                ptr.push_back(i);
                epj_link[j].init(ptr.size());
                for (CX::S32 i = 0; i < ptr.size(); i++)
                    epj_link[j].setLink(i, ptr[i]);
                std::sort(epj_link[j].info, epj_link[j].info + epj_link[j].n_link,
                          [](const typename TLinkInfo::Link &l, const typename TLinkInfo::Link &r)
                              -> bool
                          { return l.target < r.target; });
            }
        }
    }
};
class SetIndegree
{
private:
    const CX::S32 indegree;
    const Multapses multapses;
    const Autapses autapses;

public:
    SetIndegree(const CX::S32 _indegree, const Multapses _multapses, const Autapses _autapses)
        : indegree(_indegree),
          multapses(_multapses),
          autapses(_autapses){};
    template <class TLinkInfo, class TPost, class TSpike>
    inline void operator()(std::vector<TLinkInfo> &epj_link,
                           std::vector<TPost> &epi_org,
                           const std::vector<TSpike> &spk_tot)
    {
        if (epi_org.size() == 0)
            return;
        const CX::S32 n_threads = CX::Comm::getNumberOfThread();
        const CX::S32 n_epi = epi_org.size();
        const CX::S32 n_epj = epj_link.size();
        std::vector<CX::S32> link_num(n_epj, 0);
        std::vector<CX::S32> ptr(n_epj, 0);
#ifdef CORTEX_THREAD_PARALLEL
        std::vector<omp_lock_t> lock(n_epj);
        for (CX::S32 j = 0; j < n_epj; j++)
            omp_init_lock(&(lock[j]));
#endif
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel
#endif
        {
            const CX::S32 ith = CX::Comm::getThreadNum();
            std::vector<CX::S32> link_num_ith(n_epj, 0);
            std::uniform_int_distribution<CX::S32> distr(0, n_epj - 1);
            std::default_random_engine test(epi_org[0].randSeed);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
            for (CX::S32 i = 0; i < n_epi; i++)
            {
                std::default_random_engine eng(epi_org[i].randSeed);
                for (CX::S32 j = 0; j < indegree; j++)
                {
                    CX::S32 adr = distr(eng);
                    if (autapses == Autapses::Autapses_NO)
                        while (adr == epi_org[i].id)
                            adr = distr(eng);
                    link_num_ith[adr]++;
                }
            }
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp critical
#endif
            for (CX::S32 j = 0; j < n_epj; j++)
                link_num[j] += link_num_ith[j];
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp barrier // important
#endif
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
            for (CX::S32 j = 0; j < n_epj; j++)
                epj_link[j].init(link_num[j]);
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
            for (CX::S32 i = 0; i < n_epi; i++)
            {
                std::default_random_engine eng(epi_org[i].randSeed);
                for (CX::S32 j = 0; j < indegree; j++)
                {
                    CX::S32 adr = distr(eng);
                    if (autapses == Autapses::Autapses_NO)
                        while (adr == epi_org[i].id)
                            adr = distr(eng);
#ifdef CORTEX_THREAD_PARALLEL
                    omp_set_lock(&(lock[adr]));
#endif
                    epj_link[adr].setLink(ptr[adr], i);
                    ptr[adr]++;
#ifdef CORTEX_THREAD_PARALLEL
                    omp_unset_lock(&(lock[adr]));
#endif
                }
            }
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp for
#endif
            for (CX::S32 j = 0; j < n_epj; j++)
                std::sort(epj_link[j].info, epj_link[j].info + epj_link[j].n_link,
                          [](const typename TLinkInfo::Link &l, const typename TLinkInfo::Link &r)
                              -> bool
                          { return l.target < r.target; });
        } // end omp
    }
};
class SetWeightFixed
{
private:
    const CX::F64 weight;

public:
    SetWeightFixed(const CX::F64 _weight)
        : weight(_weight){};
    template <class TLinkInfo, class TPost, class TSpike>
    inline void operator()(std::vector<TLinkInfo> &epj_link,
                           std::vector<TPost> &epi_org,
                           const std::vector<TSpike> &spk_tot)
    {
        if (epi_org.size() == 0)
            return;
        const CX::S32 n_epj = epj_link.size();
#ifdef CORTEX_THREAD_PARALLEL
#pragma omp parallel for
#endif
        for (CX::S32 j = 0; j < n_epj; j++)
            for (CX::S32 i = 0; i < epj_link[j].n_link; i++)
                epj_link[j].setWeight(i, weight);
    }
};