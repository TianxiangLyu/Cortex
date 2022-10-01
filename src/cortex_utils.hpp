#pragma once
#include <unistd.h>
namespace CX = Cortex;
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