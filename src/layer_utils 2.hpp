namespace Cortex
{
    class FileHeader
    {
    public:
        S32 numOfNeuron;
        F64 time;
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
        S32 ret;
        char dir_name[256];
        sprintf(dir_name, "result/%s", layer_name.c_str());
        // mkdir result
        if (Comm::getRank() == 0)
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
        Comm::broadcast(&ret, 1);
        if (ret != 0)
        {
            if (Comm::getRank() == 0)
                fprintf(stderr, "Directory %s fails to be made.\n", "result");
            Abort();
        }
        // mkdir result/LayerName
        if (Comm::getRank() == 0)
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
        Comm::broadcast(&ret, 1);
        if (ret != 0)
        {
            if (Comm::getRank() == 0)
                fprintf(stderr, "Directory %s fails to be made.\n", dir_name);
            Abort();
        }
    }
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