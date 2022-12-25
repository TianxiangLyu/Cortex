#include <cortex_utils.hpp>
inline CX::F32 GaussianDistribution2D(CX::F32 dx, CX::F32 dy,
                                      CX::F32 P_center,
                                      CX::F32 sigma_x, CX::F32 sigma_y,
                                      CX::F32 mean_x, CX::F32 mean_y,
                                      CX::F32 rho, CX::F32 c)
{
    const CX::F32 A = (dx - mean_x) * (dx - mean_x) / (sigma_x * sigma_x);
    const CX::F32 B = (dy - mean_y) * (dy - mean_y) / (sigma_y * sigma_y);
    const CX::F32 C = 2 * rho * (dx - mean_x) * (dy - mean_y) / (sigma_x * sigma_y);
    const CX::F32 D = 2 * (1 - rho * rho);
    return c + P_center * expf(-(A - B + C) / D);
}
template <class Tfunc_fx>
CX::F64 bisectionSearch(Tfunc_fx pfunc_fx,
                        const CX::F64 tolerance,
                        CX::F64 x_low,
                        CX::F64 x_high)
{
    CX::F64 fx_low = pfunc_fx(x_low);
    CX::F64 fx_high = pfunc_fx(x_high);
    if (fx_low == 0)
        return x_low;
    if (fx_high == 0)
        return x_high;
    if (fx_low * fx_high > 0) // sethighest
    {
        CORTEX_PRINT_ERROR("bisectionSearch Error! fx_low * fx_high > 0");
        CX::Abort(-1);
    }
    if (x_low > x_high)
    {
        CORTEX_PRINT_ERROR("bisectionSearch Error! x_low > x_high");
        CX::Abort(-1);
    }
    while ((x_high - x_low) / 2 > tolerance)
    {
        CX::F64 x_mid = (x_low + x_high) / 2;
        CX::F64 fx_mid = pfunc_fx(x_mid);
        if (fx_mid == 0)
            break;
        if (fx_low * fx_mid < 0)
        {
            x_high = x_mid;
            fx_high = fx_mid;
        }
        else
        {
            x_low = x_mid;
            fx_low = fx_mid;
        }
    }
    return (x_low + x_high) / 2;
}
template <class Tfunc_fx>
CX::F64 bisectionSearchHighOff(Tfunc_fx pfunc_fx,
                               const CX::F64 tolerance,
                               CX::F64 x_low,
                               CX::F64 x_high)
{
    CX::F64 fx_low = pfunc_fx(x_low);
    CX::F64 fx_high = pfunc_fx(x_high);
    if (fx_low == 0)
        return x_low;
    if (fx_high == 0)
        return x_high;
    if (fx_low * fx_high > 0) // sethighest
    {
        return x_high;
    }
    if (x_low > x_high)
    {
        CORTEX_PRINT_ERROR("bisectionSearch Error! x_low > x_high");
        CX::Abort(-1);
    }
    while ((x_high - x_low) / 2 > tolerance)
    {
        CX::F64 x_mid = (x_low + x_high) / 2;
        CX::F64 fx_mid = pfunc_fx(x_mid);
        if (fx_mid == 0)
            break;
        if (fx_low * fx_mid < 0)
        {
            x_high = x_mid;
            fx_high = fx_mid;
        }
        else
        {
            x_low = x_mid;
            fx_low = fx_mid;
        }
    }
    return (x_low + x_high) / 2;
}
class getDeviationDistrUniform2D
{
private:
    const CX::F64 x_low_;
    const CX::F64 x_high_;
    const CX::F64 y_low_;
    const CX::F64 y_high_;
    const CX::S64 num_;

public:
    getDeviationDistrUniform2D(const CX::F64 x_low,
                               const CX::F64 x_high,
                               const CX::F64 y_low,
                               const CX::F64 y_high,
                               const CX::S64 num)
        : x_low_(x_low),
          x_high_(x_high),
          y_low_(y_low),
          y_high_(y_high),
          num_(num){};
    CX::S64 operator()(const CX::F64 delta)
    {
        CX::S64 numPtcl = 0;
        for (CX::F64 x = delta / 2; x < x_high_; x += delta)
            for (CX::F64 y = delta / 2; y < y_high_; y += delta)
                numPtcl++;
        for (CX::F64 x = -delta / 2; x > x_low_; x -= delta)
            for (CX::F64 y = delta / 2; y < y_high_; y += delta)
                numPtcl++;
        for (CX::F64 x = -delta / 2; x > x_low_; x -= delta)
            for (CX::F64 y = -delta / 2; y > y_low_; y -= delta)
                numPtcl++;
        for (CX::F64 x = delta / 2; x < x_high_; x += delta)
            for (CX::F64 y = -delta / 2; y > y_low_; y -= delta)
                numPtcl++;
        return num_ - numPtcl;
    }
};
class DistrEqualNullPos
{
private:
    const CX::S64 num_;

public:
    DistrEqualNullPos(const CX::S64 num)
        : num_(num){};
    template <class Tneu>
    CX::F64ort operator()(CX::NeuronInstance<Tneu> &neuron)
    {
        if (neuron.getCommInfo().isCommNull())
            return CX::F64ort(CX::F64vec(0), 1);
        const CX::S32 rank = neuron.getCommInfo().getRank();
        const CX::S32 n_proc = neuron.getCommInfo().getNumberOfProc();
        const CX::S32 n_tot = num_ % n_proc == 0 ? num_ : (num_ / n_proc + 1) * n_proc;
        const CX::S64 n_loc = n_tot / n_proc;
        neuron.setNumberOfParticleLocal(n_loc);
        for (CX::S32 i = 0; i < neuron.getNumberOfParticleLocal(); i++)
        {
            neuron[i].pos = CX::F64vec(0);
            // neuron[i].id = i + n_loc * rank;
        }
        return CX::F64ort(CX::F64vec(0), 1);
    }
};
class DistrUniform2D
{
private:
    const CX::F64vec center_;
    const CX::F64 length_;
    const CX::S64 num_;
    const CX::F64 tolerance_;

public:
    DistrUniform2D(const CX::F64vec center,
                   const CX::F64 length,
                   const CX::S64 num,
                   const CX::F64 tolerance = 1e-9)
        : center_(center),
          length_(length),
          num_(num),
          tolerance_(tolerance){};
    template <class Tneu>
    CX::F64ort operator()(CX::NeuronInstance<Tneu> &neuron)
    {
        if (neuron.getCommInfo().isCommNull())
            return CX::F64ort(center_, length_);
        CX::F64 delta = 2 * length_ / sqrt(num_);
        CX::F64 delta_low = delta;
        CX::F64 delta_high = delta;
        CX::S64 numPtclGlobal = num_;
        CX::F64 x_low = -length_ + center_.x;
        CX::F64 x_high = length_ + center_.x;
        CX::F64 y_low = -length_ + center_.y;
        CX::F64 y_high = length_ + center_.y;
        CX::S32 id = 0;
        CX::S32 i = 0;
        CX::S64 deviation = getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_).operator()(delta);
#ifdef CORTEX_DEBUG_PRINT
        if (neuron.getCommInfo().getRank() == 0)
        {
            std::cout << "The deviation is  " << deviation << std::endl;
            std::cout << "The number of neurons is  " << numPtclGlobal << std::endl;
        }
#endif
        if (deviation < 0)
        {
            while (getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_).operator()(delta_high) < 0)
                delta_high += 0.01 * delta;
            delta = bisectionSearch(getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_), tolerance_, delta_low, delta_high);
#ifdef CORTEX_DEBUG_PRINT
            if (neuron.getCommInfo().getRank() == 0)
            {
                std::cout << "deviation < 0 " << std::endl;
                std::cout << "delta_low " << delta_low << std::endl;
                std::cout << "delta_high " << delta_high << std::endl;
                std::cout << "delta " << delta << std::endl;
                std::cout << "deviation low " << getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_).operator()(delta_low) << std::endl;
                std::cout << "deviation high " << getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_).operator()(delta_high) << std::endl;
            }
#endif
        }
        else if (deviation > 0)
        {
            while (getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_).operator()(delta_low) > 0)
                delta_low -= 0.01 * delta;
            delta = bisectionSearch(getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_), tolerance_, delta_low, delta_high);
#ifdef CORTEX_DEBUG_PRINT
            if (neuron.getCommInfo().getRank() == 0)
            {
                std::cout << "deviation > 0 " << std::endl;
                std::cout << "delta_low " << delta_low << std::endl;
                std::cout << "delta_high " << delta_high << std::endl;
                std::cout << "delta " << delta << std::endl;
                std::cout << "deviation low " << getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_).operator()(delta_low) << std::endl;
                std::cout << "deviation high " << getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_).operator()(delta_high) << std::endl;
            }
#endif
        }
        deviation = getDeviationDistrUniform2D(x_low, x_high, y_low, y_high, num_).operator()(delta);
        numPtclGlobal = num_ - deviation;
        if (neuron.getCommInfo().getRank() == 0)
        {
            // std::cout << "The deviation is  " << deviation << std::endl;
            std::cout << "The number of neurons is  " << numPtclGlobal << std::endl;
        }
        const CX::S32 numPtclLocal = neuron.getCommInfo().getRank() < neuron.getCommInfo().getNumberOfProc() - 1 ? numPtclGlobal / neuron.getCommInfo().getNumberOfProc() : numPtclGlobal / neuron.getCommInfo().getNumberOfProc() + numPtclGlobal % neuron.getCommInfo().getNumberOfProc();
        neuron.setNumberOfParticleLocal(numPtclLocal);
        const CX::S32 i_head = (numPtclGlobal / neuron.getCommInfo().getNumberOfProc()) * neuron.getCommInfo().getRank();
        const CX::S32 i_tail = i_head + numPtclLocal;
        for (CX::F64 x = delta / 2; x < x_high; x += delta)
            for (CX::F64 y = delta / 2; y < y_high; y += delta)
            {
                if (i_head <= id && id <= i_tail)
                {
                    neuron[i].pos.x = x;
                    neuron[i].pos.y = y;
                    // neuron[i].id = id;
                    i++;
                }
                id++;
            }
        for (CX::F64 x = -delta / 2; x > x_low; x -= delta)
            for (CX::F64 y = delta / 2; y < y_high; y += delta)
            {
                if (i_head <= id && id <= i_tail)
                {
                    neuron[i].pos.x = x;
                    neuron[i].pos.y = y;
                    // neuron[i].id = id;
                    i++;
                }
                id++;
            }
        for (CX::F64 x = -delta / 2; x > x_low; x -= delta)
            for (CX::F64 y = -delta / 2; y > y_low; y -= delta)
            {
                if (i_head <= id && id <= i_tail)
                {
                    neuron[i].pos.x = x;
                    neuron[i].pos.y = y;
                    // neuron[i].id = id;
                    i++;
                }
                id++;
            }
        for (CX::F64 x = delta / 2; x < x_high; x += delta)
            for (CX::F64 y = -delta / 2; y > y_low; y -= delta)
            {
                if (i_head <= id && id <= i_tail)
                {
                    neuron[i].pos.x = x;
                    neuron[i].pos.y = y;
                    // neuron[i].id = id;
                    i++;
                }
                id++;
            }
        return CX::F64ort(center_, length_);
    }
};
class NeuronSpikeRandomRef
{
private:
    const CX::F32 time;
    const CX::F32 dt;
    const CX::F32 ratio;

public:
    NeuronSpikeRandomRef(const CX::F32 _time, const CX::F32 _dt, const CX::F32 _ratio)
        : time(_time),
          dt(_dt),
          ratio(_ratio){};
    template <class Tneu>
    void operator()(CX::NeuronInstance<Tneu> &neuron)
    {
        std::random_device rd;
        std::default_random_engine eng(rd());
        std::uniform_real_distribution<CX::F32> distr(0.0, 1.0);
        for (CX::S32 i = 0; i < neuron.getNumberOfParticleLocal(); i++)
        {
            if (neuron[i].CheckRef())
            {
                if (distr(eng) < ratio)
                {
                    neuron[i].GetSpike(time);
                    neuron[i].SetRef();
                }
                else
                {
                    neuron[i].NoSpike();
                }
            }
            else
            {
                neuron[i].NoSpike();
                neuron[i].UpdateRef(dt);
            }
        }
    }
};
class NeuronSpikeSet
{
private:
    const CX::S32 id_start_;
    const CX::S32 id_end_;

public:
    NeuronSpikeSet(const CX::S32 id_start, const CX::S32 id_end)
        : id_start_(id_start),
          id_end_(id_end){};
    template <class Tneu>
    void operator()(CX::NeuronInstance<Tneu> &neuron)
    {
        for (CX::S32 i = 0; i < neuron.getNumberOfParticleLocal(); i++)
            if (id_start_ <= neuron[i].id && neuron[i].id <= id_end_)
                neuron[i].spike = true;
    }
};
namespace Cortex
{
    class LayerTimeProfile
    {
    public:
        F64 spike_broadcast;
        F64 write_back_map_exc;
        F64 write_back_map_inh;
        F64 write_back_index_exc;
        F64 write_back_index_inh;
        F64 calc_dynamics;
        F64 getTotalTime() const
        {
            return spike_broadcast +
                   write_back_map_exc +
                   write_back_map_inh +
                   write_back_index_exc +
                   write_back_index_inh +
                   calc_dynamics;
        }
        void dump(std::ostream &fout = std::cout,
                  const S32 level = 0) const
        {
            fout << "total_time= " << getTotalTime() << std::endl;
            fout << "spike_broadcast= " << spike_broadcast << std::endl;
            fout << "write_back_map_exc= " << write_back_map_exc << std::endl;
            fout << "write_back_map_inh= " << write_back_map_inh << std::endl;
            fout << "write_back_index_exc= " << write_back_index_exc << std::endl;
            fout << "write_back_index_inh= " << write_back_index_inh << std::endl;
            fout << "calc_dynamics= " << calc_dynamics << std::endl;
        }
        LayerTimeProfile()
        {
            spike_broadcast = 0;
            write_back_map_exc = 0;
            write_back_map_inh = 0;
            write_back_index_exc = 0;
            write_back_index_inh = 0;
            calc_dynamics = 0;
        }

        LayerTimeProfile operator+(const LayerTimeProfile &rhs) const
        {
            LayerTimeProfile ret;
            ret.spike_broadcast = this->spike_broadcast + rhs.spike_broadcast;
            ret.write_back_map_exc = this->write_back_map_exc + rhs.write_back_map_exc;
            ret.write_back_map_inh = this->write_back_map_inh + rhs.write_back_map_inh;
            ret.write_back_index_exc = this->write_back_index_exc + rhs.write_back_index_exc;
            ret.write_back_index_inh = this->write_back_index_inh + rhs.write_back_index_inh;
            ret.calc_dynamics = this->calc_dynamics + rhs.calc_dynamics;
            return ret;
        }

        const LayerTimeProfile &operator+=(const LayerTimeProfile &rhs)
        {
            (*this) = (*this) + rhs;
            return (*this);
        }

        void clear()
        {
            spike_broadcast = 0;
            write_back_map_exc = 0;
            write_back_map_inh = 0;
            write_back_index_exc = 0;
            write_back_index_inh = 0;
            calc_dynamics = 0;
        }
    };
}