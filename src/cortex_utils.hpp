#pragma once
#include <unistd.h>
namespace CX = Cortex;
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