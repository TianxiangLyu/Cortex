#include <domain_info.hpp>
#include <dst_domain_info.hpp>
#include <cortex_system.hpp>
#include <random>
#include <iomanip>
#include <iostream>
#include <memory>
#include <atomic>
#include <thread>
#include <sys/stat.h>
#include <vector>
#include <algorithm>
#include <delay_queue.hpp>
#include <delay_rma_queue.hpp>
#include <population_utils.hpp>
namespace Cortex
{
    static std::unordered_map<std::string, S32> name_to_population_id_glb_;
    static S32 num_population_glb_ = 0;
    template <class Tneu,
              class Tspk>
    class PopulationInfo // need a new API for system information dump
    {
    private:
        S32 population_id_;
        std::string population_name_;

    public:
        LayerTimeProfile time_profile;
        DomainInfo dinfo_;
        NeuronInstance<Tneu> neuron_;
        DelayQueue<Tspk> queue_;
        // DelayRmaQueue<Tspk> rma_queue_;
        DstDomainInfo dst_dinfo_;
        std::vector<Tspk> spk_tot_;
        ~PopulationInfo(){};
        PopulationInfo(){};
        Tneu &operator[](const int i) { return neuron_[i]; }
        template <class Tfunc_init>
        void initialize(const std::string population_name,
                        const BOUNDARY_CONDITION bc,
                        Tfunc_init func_init,
                        MPI_Group group)
        {
            dinfo_.initGroup(group);
            neuron_.initGroup(group);
            dst_dinfo_.initGroup(group);
            population_name_ = population_name;
            population_id_ = num_population_glb_++;
            checkLayerName(population_name); // couldn't ganrantee thread safe here
            dinfo_.setBoundaryCondition(bc);
            const F64ort pos_root_domain = func_init(neuron_);
            dinfo_.setPosRootDomain(pos_root_domain.low_, pos_root_domain.high_); // useless for BOUNDARY_OPEN
            dinfo_.decomposeDomainAll(neuron_);
            neuron_.exchangeParticle(dinfo_); // set comm_info_;
            std::random_device rd;
            std::default_random_engine eng{rd()};
            std::uniform_int_distribution<S32> distr(1, 4 * neuron_.getNumberOfParticleGlobal());
            for (S32 i = 0; i < neuron_.getNumberOfParticleLocal(); i++)
            {
                const S32 rand_seed = distr(eng);
                neuron_[i].init(rand_seed);
            }
            setNeuronID();
            dst_dinfo_.setSrcInfo(dinfo_);
            const S64 n_tot = getNumGlobal();
            if (Comm::getRank() == 0)
                std::cout << "Population " << population_name << " Initialization " << n_tot << std::endl;
        }
        template <class Tfunc_init>
        PopulationInfo(const std::string population_name,
                       const BOUNDARY_CONDITION bc,
                       Tfunc_init func_init) // should return the pos root domain
        {
            MPI_Group world_group;
            MPI_Comm_group(MPI_COMM_WORLD, &world_group);
            initialize(population_name, bc, func_init, world_group);
        };
        template <class Tfunc_init>
        PopulationInfo(const std::string population_name,
                       const BOUNDARY_CONDITION bc,
                       Tfunc_init func_init,
                       MPI_Group group) // should return the pos root domain
        {
            initialize(population_name, bc, func_init, group);
        };
        S32 getLayerID() { return population_id_; }
        std::string getLayerName() { return population_name_; }
        S64 getNumLocal() { return neuron_.getNumberOfParticleLocal(); }
        S64 getNumGlobal() { return neuron_.getNumberOfParticleGlobal(); }
        Tneu *data(const S32 id = 0) const { return neuron_.getParticlePointer(id); }
        DomainInfo &getDinfo() { return dinfo_; }
        void checkLayerName(const std::string population_name)
        {
            auto search = name_to_population_id_glb_.find(population_name);
            if (search == name_to_population_id_glb_.end())
            {
                name_to_population_id_glb_.insert(std::pair<std::string, S32>(population_name, getLayerID()));
            }
            else
            {
                std::cerr << "Population name " << population_name << " has been existed " << std::endl;
                std::cerr << "Exist ID " << search->second << " this ID " << getLayerID() << std::endl;
                Abort(-1);
            }
        }
        void freeSpkAll()
        {
            spk_tot_.clear();
            spk_tot_.shrink_to_fit();
            Comm::barrier();
        }
        void SpikeUpdate(const F64 time)
        {
            queue_.PopSpk(time, dst_dinfo_);
            queue_.PushSpk(time, neuron_, dst_dinfo_);
            queue_.SyncAll(time, dst_dinfo_);
            /* if (Comm::getRank() == 0)
                queue_.dump(time, dst_dinfo_); */
        }
        void initRMA()
        {
            // rma_queue_.initialize();
        }
        void freeRMA()
        {
            // rma_queue_.free();
        }
        void SpkAllGather()
        {
            const S32 n_loc = neuron_.getNumberOfParticleLocal();
            if (dst_dinfo_.getCommInfo().isNotCommNull())
            {
                std::vector<Tspk> spk_loc;
                spk_loc.reserve(n_loc);
                for (S32 i = 0; i < n_loc; i++)
                    spk_loc.push_back(Tspk(neuron_[i]));
                dst_dinfo_.getCommInfo().allGatherVAll(spk_loc, spk_loc.size(), spk_tot_);
            }
            Comm::barrier();
        }
        void setNeuronID()
        {
            if (dinfo_.getCommInfo().isNotCommNull())
            {
                const S32 rank = dinfo_.getCommInfo().getRank();
                const S32 n_proc = dinfo_.getCommInfo().getNumberOfProc();
                const S32 n_loc = neuron_.getNumberOfParticleLocal();
                std::vector<S32> n_recv(n_proc, 0);
                std::vector<S32> n_disp(n_proc, 0);
                dinfo_.getCommInfo().allGather(&n_loc, 1, n_recv.data());
                for (S32 i = 1; i < n_proc; i++)
                    n_disp[i] = n_recv[i - 1] + n_disp[i - 1];
                /* if(Comm::getRank() == 0)
                {
                    for(S32 i = 0; i < n_proc; i++)
                        std::cout<<n_disp[i]<<" ";
                    std::cout<<std::endl;
                } */
                for (S32 i = 0; i < neuron_.getNumberOfParticleLocal(); i++)
                    neuron_[i].id = n_disp[rank] + i;
            }
        }
        void addConn(const S32 target, const F64 delay, DomainInfo &d)
        {
            dst_dinfo_.addDstInfo(target, delay, d);
            queue_.AddTarget(target, delay);

            /* if (Comm::getRank() == 0)
                dst_dinfo_.dump(); */
        }
        template <class Tfunc_stream>
        void operator()(Tfunc_stream pfunc_stream)
        {
            pfunc_stream(neuron_);
        }
        template <class Tfunc_dyn>
        void NeuralDynamics(Tfunc_dyn pfunc_dyn)
        {
            const F64 time_offset = GetWtime();
            pfunc_dyn(neuron_);
            time_profile.calc_dynamics += GetWtime() - time_offset;
        }
    };
    template <typename Tntyp>
    class Population
    {
    public:
        typedef PopulationInfo<typename Tntyp::Neuron,
                               typename Tntyp::Spike>
            Default;
    };
}