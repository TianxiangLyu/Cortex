#pragma once
#include <unistd.h>
namespace CX = Cortex;
CX::F64 propagator_31(CX::F64 tau_syn, CX::F64 tau, CX::F64 C, CX::F64 h)
{
    const CX::F64 P31_linear = 1 / (3. * C * tau * tau) * h * h * h * (tau_syn - tau) * std::exp(-h / tau);
    const CX::F64 P31 = 1 / C * (std::exp(-h / tau_syn) * std::expm1(-h / tau + h / tau_syn) / (tau / tau_syn - 1) * tau - h * std::exp(-h / tau_syn)) / (-1 - -tau / tau_syn) * tau;
    const CX::F64 P31_singular = h * h / 2 / C * std::exp(-h / tau);
    const CX::F64 dev_P31 = std::abs(P31 - P31_singular);

    if (tau == tau_syn or (std::abs(tau - tau_syn) < 0.1 and dev_P31 > 2 * std::abs(P31_linear)))
    {
        return P31_singular;
    }
    else
    {
        return P31;
    }
}
CX::F64 propagator_32(CX::F64 tau_syn, CX::F64 tau, CX::F64 C, CX::F64 h)
{
    const CX::F64 P32_linear = 1 / (2. * C * tau * tau) * h * h * (tau_syn - tau) * std::exp(-h / tau);
    const CX::F64 P32_singular = h / C * std::exp(-h / tau);
    const CX::F64 P32 = -tau / (C * (1 - tau / tau_syn)) * std::exp(-h / tau_syn) * std::expm1(h * (1 / tau_syn - 1 / tau));

    const CX::F64 dev_P32 = std::abs(P32 - P32_singular);

    if (tau == tau_syn or (std::abs(tau - tau_syn) < 0.1 and dev_P32 > 2 * std::abs(P32_linear)))
    {
        return P32_singular;
    }
    else
    {
        return P32;
    }
}
template <class Tlink>
inline CX::S32 LFsearch(Tlink *info, CX::S32 num, CX::S32 target)
{
    CX::S32 lo = 0;
    CX::S32 hi = num - 1;
    while (hi - lo > 1)
    {
        CX::S32 mid = (hi + lo) / 2;
        if (info[mid].target < target)
            lo = mid + 1;
        else
            hi = mid;
    }
    if (info[lo].target >= target)
        return lo;
    else
        return hi;
}
template <class Tlink>
inline CX::S32 RHsearch(Tlink *info, CX::S32 num, CX::S32 target)
{
    CX::S32 lo = 0;
    CX::S32 hi = num - 1;
    while (hi - lo > 1)
    {
        CX::S32 mid = (hi + lo) / 2;
        if (info[mid].target > target)
            hi = mid - 1;
        else
            lo = mid;
    }
    if (info[hi].target <= target)
        return hi;
    else
        return lo;
}
class FileHeader
{
public:
    CX::S32 numOfNeuron;
    CX::F64 time;
    int readAscii(FILE *fp)
    {
        fscanf(fp, "%lf\n", &time);
        fscanf(fp, "%d\n", &numOfNeuron);
        return numOfNeuron;
    }
    void writeAscii(FILE *fp) const
    {
        fprintf(fp, "%e\n", time);
        fprintf(fp, "%d\n", numOfNeuron);
    }
};
void makeOutputDirectory(std::string layer_name)
{
    struct stat st;
    CX::S32 ret;
    char dir_name[256];
    sprintf(dir_name, "result/%s", layer_name.c_str());
    // mkdir result
    if (CX::Comm::getRank() == 0)
    {
        if (stat("result", &st) != 0)
        {
            ret = mkdir("result", 0777);
        }
        else
        {
            ret = 0; // the directory named dir_name already exists.
        }
    }
    CX::Comm::broadcast(&ret, 1);
    if (ret != 0)
    {
        if (CX::Comm::getRank() == 0)
            fprintf(stderr, "Directory %s fails to be made.\n", "result");
        CX::Abort();
    }
    // mkdir result/LayerName
    if (CX::Comm::getRank() == 0)
    {
        if (stat(dir_name, &st) != 0)
        {
            ret = mkdir(dir_name, 0777);
        }
        else
        {
            ret = 0; // the directory named dir_name already exists.
        }
    }
    CX::Comm::broadcast(&ret, 1);
    if (ret != 0)
    {
        if (CX::Comm::getRank() == 0)
            fprintf(stderr, "Directory %s fails to be made.\n", dir_name);
        CX::Abort();
    }
}