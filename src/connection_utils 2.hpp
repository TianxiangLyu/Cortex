#include <cortex.hpp>
namespace CX = Cortex;
class LinkPseudoRandomEPIGauss2D
{
private:
    const CX::F32 sigma_x;
    const CX::F32 sigma_y;
    const CX::F32 mean_x;
    const CX::F32 mean_y;
    const CX::F32 rho;
    const CX::F32 c;
    const CX::F32 inv_sigma_x2;
    const CX::F32 inv_sigma_y2;
    const CX::F32 inv_sigma_xy;

public:
    LinkPseudoRandomEPIGauss2D(CX::F32 _sigma_x, CX::F32 _sigma_y,
                               CX::F32 _mean_x, CX::F32 _mean_y,
                               CX::F32 _rho, CX::F32 _c)
        : sigma_x(_sigma_x),
          sigma_y(_sigma_y),
          mean_x(_mean_x),
          mean_y(_mean_y),
          rho(_rho),
          c(_c),
          inv_sigma_x2(1.0 / (_sigma_x * _sigma_x)),
          inv_sigma_y2(1.0 / (_sigma_y * _sigma_y)),
          inv_sigma_xy(1.0 / (_sigma_x * _sigma_y)){};
    template <class Tepi, class Tepj>
    CX::S32 operator()(const Tepi &ep_i,
                    const Tepj &ep_j)
    {
        const CX::F32 dx = fabs(ep_j.pos.x - ep_i.pos.x);
        const CX::F32 dy = fabs(ep_j.pos.y - ep_i.pos.y);
        const CX::F32 r2 = dx * dx + dy * dy;
        const CX::F32 A = (dx - mean_x) * (dx - mean_x) / (sigma_x * sigma_x);
        const CX::F32 B = (dy - mean_y) * (dy - mean_y) / (sigma_y * sigma_y);
        const CX::F32 C = 2 * rho * (dx - mean_x) * (dy - mean_y) / (sigma_x * sigma_y);
        const CX::F32 D = 2 * (1 - rho * rho);
        const CX::F32 gauss2D = c + ep_i.peak * expf(-(A - B + C) / D);
        const CX::S32 combined_seed = ep_i.randSeed * ep_j.randSeed + 12345;
        const CX::S32 random_connection = (combined_seed / 65536) % 32768; // RandMax assumed to be 32767
        return (r2 <= ep_i.Rsearch * ep_i.Rsearch && gauss2D * 32767 >= random_connection);
    }
};
class LinkPseudoRandomEPIGauss2D_peak
{
private:
    const CX::F32 sigma_x;
    const CX::F32 sigma_y;
    const CX::F32 mean_x;
    const CX::F32 mean_y;
    const CX::F32 rho;
    const CX::F32 c;
    const CX::F32 inv_sigma_x2;
    const CX::F32 inv_sigma_y2;
    const CX::F32 inv_sigma_xy;

public:
    LinkPseudoRandomEPIGauss2D_peak(CX::F32 _sigma_x, CX::F32 _sigma_y,
                                    CX::F32 _mean_x, CX::F32 _mean_y,
                                    CX::F32 _rho, CX::F32 _c)
        : sigma_x(_sigma_x),
          sigma_y(_sigma_y),
          mean_x(_mean_x),
          mean_y(_mean_y),
          rho(_rho),
          c(_c),
          inv_sigma_x2(1.0 / (_sigma_x * _sigma_x)),
          inv_sigma_y2(1.0 / (_sigma_y * _sigma_y)),
          inv_sigma_xy(1.0 / (_sigma_x * _sigma_y)){};
    template <class Tepi, class Tepj>
    CX::S32 operator()(const Tepi &ep_i,
                    const Tepj &ep_j,
                    const CX::F32 peak)
    {
        const CX::F32 dx = fabs(ep_j.pos.x - ep_i.pos.x);
        const CX::F32 dy = fabs(ep_j.pos.y - ep_i.pos.y);
        const CX::F32 r2 = dx * dx + dy * dy;
        const CX::F32 A = (dx - mean_x) * (dx - mean_x) / (sigma_x * sigma_x);
        const CX::F32 B = (dy - mean_y) * (dy - mean_y) / (sigma_y * sigma_y);
        const CX::F32 C = 2 * rho * (dx - mean_x) * (dy - mean_y) / (sigma_x * sigma_y);
        const CX::F32 D = 2 * (1 - rho * rho);
        const CX::F32 gauss2D = c + peak * expf(-(A - B + C) / D);
        const CX::S32 combined_seed = ep_i.randSeed * ep_j.randSeed + 12345;
        const CX::S32 random_connection = (combined_seed / 65536) % 32768; // RandMax assumed to be 32767
        return (r2 <= ep_i.Rsearch * ep_i.Rsearch && gauss2D * 32767 >= random_connection);
    }
};
class LinkIDG_peak
{
public:
    template <class Tepi, class Tepj>
    CX::S32 operator()(const Tepi &ep_i,
                    const Tepj &ep_j,
                    const CX::F64 peak)
    {
        if(peak == 0)
            return 0;
        const CX::F32 dx = ep_j.pos.x - ep_i.pos.x;
        const CX::F32 dy = ep_j.pos.y - ep_i.pos.y;
        const CX::F32 r2 = dx * dx + dy * dy;
        const CX::S32 combined_seed = ep_i.randSeed * ep_j.randSeed + 12345;
        const CX::F64 random_connection = (combined_seed / 65536) % 32768; // RandMax assumed to be 32767
        return (r2 <= ep_i.Rsearch * ep_i.Rsearch && peak * (CX::F64)32767.0 >= random_connection);
    }
};
class connectionEstimateIDG
{
private:
    const CX::S32 indegree_;
public:
    connectionEstimateIDG(const CX::S32 indegree)
        : indegree_(indegree){};
    template <typename Tepi, typename Tepj>
    CX::U64 operator()(const Tepi *const ep_i, const CX::S32 Nip,
                       const Tepj *const ep_j, const CX::S32 Njp,
                       const typename CX::F32 *const peak)
    {
        return Nip * indegree_;
    }
};
class ConnectionEstimateIDG
{
private:
    const CX::S32 indegree_;
public:
    ConnectionEstimateIDG(const CX::S32 indegree)
        : indegree_(indegree){};
    template <typename Tepi, typename Tepj>
    CX::S32 operator()(const Tepi *const ep_i, const CX::S32 Nip,
                       const Tepj *const ep_j, const CX::S32 Njp)
    {
        return Nip * indegree_;
    }
};
template <class Tfunc_link, class Tfunc_weight>
class WeightDefaultLink
{
private:
    Tfunc_link isLink;
    Tfunc_weight getWeight;

public:
    WeightDefaultLink(Tfunc_link _isLink,
                      Tfunc_weight _getWeight)
        : isLink(_isLink),
          getWeight(_getWeight){};
    WeightDefaultLink(){};
    template <class Tepi, class Tepj>
    void operator()(Tepi *const ep_i, const CX::S32 Nip,
                    Tepj *const ep_j, const CX::S32 Njp)
    {
        for (CX::S32 j = 0; j < Njp; j++)
        {
            ep_j[j].num_link = 0;
            for (CX::S32 i = 0; i < Nip; i++)
                if (isLink(ep_i[i], ep_j[j]))
                {
                    ep_j[j].num_link++;
                    ep_j[j].link.push_back(i);
                    ep_j[j].weight.push_back(getWeight(ep_i[i], ep_j[j]));
                }
        }
    }
};
template <class Tfunc_link, class Tfunc_weight>
class WeightPseudoRandom
{
private:
    Tfunc_link isLink;
    Tfunc_weight getWeight;

public:
    WeightPseudoRandom(Tfunc_link _isLink,
                       Tfunc_weight _getWeight)
        : isLink(_isLink),
          getWeight(_getWeight){};
    WeightPseudoRandom(){};
    template <class Tepi, class Tepj>
    void operator()(Tepi *const ep_i, const CX::S32 Nip,
                    Tepj *const ep_j, const CX::S32 Njp)
    {
        for (CX::S32 j = 0; j < Njp; j++)
        {
            ep_j[j].num_link = 0;
            for (CX::S32 i = 0; i < Nip; i++)
                ep_j[j].num_link += isLink(ep_i[i], ep_j[j]);
            ep_j[j].weight.resizeNoInitialize(ep_j[j].num_link);
            CX::S32 ptr = 0;
            for (CX::S32 i = 0; i < Nip; i++)
                if (isLink(ep_i[i], ep_j[j]))
                    ep_j[j].weight[ptr++] = getWeight(ep_i[i], ep_j[j]);
        }
    }
};
template <class Tepi, class Tepj, class Tfunc_link_peak>
class CheckConnectionEPI_peak
{
private:
    Tfunc_link_peak isLink_peak;
    const Tepi &ep_i;
    const Tepj *const ep_j;
    const CX::S32 Njp;

public:
    CheckConnectionEPI_peak(const Tepi &_epi,
                            const Tepj *const _ep_j,
                            const CX::S32 _Njp,
                            Tfunc_link_peak _isLink_peak)
        : ep_i(_epi),
          ep_j(_ep_j),
          Njp(_Njp),
          isLink_peak(_isLink_peak){};
    CX::F64 operator()(const CX::F64 peak)
    {
        CX::S32 total_connection = 0;
        if(peak == 0)
            return (CX::F64)(0 - ep_i.indegree);
        for (CX::S32 j = 0; j < Njp; j++)
            total_connection += isLink_peak(ep_i, ep_j[j], peak);
        return (CX::F64)(total_connection - ep_i.indegree);
    }
};
template <class Tepi, class Tepj, class Tfunc_link_peak>
class CountConnectionEPI_peak
{
private:
    Tfunc_link_peak isLink_peak;
    const Tepi &ep_i;
    const Tepj *const ep_j;
    const CX::S32 Njp;

public:
    CountConnectionEPI_peak(const Tepi &_epi,
                            const Tepj *const _ep_j,
                            const CX::S32 _Njp,
                            Tfunc_link_peak _isLink_peak)
        : ep_i(_epi),
          ep_j(_ep_j),
          Njp(_Njp),
          isLink_peak(_isLink_peak){};
    CX::S32 operator()(const CX::F64 peak)
    {
        CX::S32 total_connection = 0;
        for (CX::S32 j = 0; j < Njp; j++)
            total_connection += isLink_peak(ep_i, ep_j[j], peak);
        return total_connection;
    }
};
template <class Tfunc_link_peak>
class BisectionSearchAllEPI_peak
{
private:
    Tfunc_link_peak isLink_peak;
    const CX::F64 low;
    const CX::F64 high;
    const CX::F64 eps;

public:
    BisectionSearchAllEPI_peak(const CX::F32 _low,
                               const CX::F32 _high,
                               const CX::F32 _eps,
                               Tfunc_link_peak _isLink_peak)
        : low(_low),
          high(_high),
          eps(_eps),
          isLink_peak(_isLink_peak){};
    template <class Tepi, class Tepj>
    void operator()(Tepi *const ep_i, const CX::S32 Nip,
                    Tepj *const ep_j, const CX::S32 Njp)
    {
        for (CX::S32 i = 0; i < Nip; i++)
        {
            CheckConnectionEPI_peak<Tepi, Tepj, Tfunc_link_peak> pfunc_check_peak(ep_i[i], ep_j, Njp, isLink_peak);
            ep_i[i].peak = bisectionSearch(pfunc_check_peak, eps, low, high);
        }
    }
};