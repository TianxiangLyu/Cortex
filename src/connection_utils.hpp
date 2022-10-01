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