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
#include <layer_utils.hpp>
namespace Cortex
{
    static std::unordered_map<std::string, S32> name_to_layer_id_glb_;
    static S32 num_layer_glb_ = 0;
    template <class Tneu,
              class Tspk>
    class LayerInfo // need a new API for system information dump
    {
    private:
        const S32 layer_id_;
        const std::string layer_name_;

    public:
        LayerTimeProfile time_profile;
        DomainInfo dinfo_;
        NeuronInstance<Tneu> neuron_;
        DelayQueue<Tspk> queue_;
        DelayRmaQueue<Tspk> rma_queue_;
        DstDomainInfo dst_dinfo_;
        std::vector<Tspk> spk_tot_;
        ~LayerInfo(){};
        LayerInfo(const LayerInfo &);
        Tneu &operator[](const int i) { return neuron_[i]; }
        const std::vector<Tspk> &geTneuall() const { return spk_tot_; }
        template <class Tfunc_init>
        LayerInfo(const std::string layer_name,
                  const BOUNDARY_CONDITION bc,
                  Tfunc_init func_init) // should return the pos root domain
            : dinfo_(),
              neuron_(),
              layer_name_(layer_name),
              layer_id_(num_layer_glb_++),
              rma_queue_(dst_dinfo_)
        {
            if (Comm::getRank() == 0)
                std::cout << "Layer " << layer_name << " Initialization" << std::endl;
            checkLayerName(layer_name); // couldn't ganrantee thread safe here
            dinfo_.setBoundaryCondition(bc);
            const F64ort pos_root_domain = func_init(neuron_);
            dinfo_.setPosRootDomain(pos_root_domain.low_, pos_root_domain.high_); // useless for BOUNDARY_OPEN
            dinfo_.decomposeDomainAll(neuron_);
            neuron_.exchangeParticle(dinfo_);
            for (S32 i = 0; i < neuron_.getNumberOfParticleLocal(); i++)
                neuron_[i].init(rand());
            dst_dinfo_.setSrcInfo(dinfo_);
            initWorld();
        };
        template <class Tfunc_init>
        LayerInfo(const std::string layer_name,
                  const BOUNDARY_CONDITION bc,
                  Tfunc_init func_init,
                  MPI_Group group) // should return the pos root domain
            : dinfo_(group),
              neuron_(group),
              layer_name_(layer_name),
              layer_id_(num_layer_glb_++),
              rma_queue_(dst_dinfo_)
        {
            if (Comm::getRank() == 0)
                std::cout << "Layer " << layer_name << " Initialization" << std::endl;
            checkLayerName(layer_name); // couldn't ganrantee thread safe here
            dinfo_.setBoundaryCondition(bc);
            const F64ort pos_root_domain = func_init(neuron_);
            dinfo_.setPosRootDomain(pos_root_domain.low_, pos_root_domain.high_); // useless for BOUNDARY_OPEN
            dinfo_.decomposeDomainAll(neuron_);
            neuron_.exchangeParticle(dinfo_); // set comm_info_;
            for (S32 i = 0; i < neuron_.getNumberOfParticleLocal(); i++)
                neuron_[i].init(rand());
            dst_dinfo_.setSrcInfo(dinfo_);
            initWorld();
        };
        S32 getLayerID() { return layer_id_; }
        std::string getLayerName() { return layer_name_; }
        S64 getNumLocal() { return neuron_.getNumberOfParticleLocal(); }
        S64 getNumGlobal() { return neuron_.getNumberOfParticleGlobal(); }
        Tneu *data(const S32 id = 0) const { return neuron_.getParticlePointer(id); }
        DomainInfo &getDinfo() { return dinfo_; }
        void checkLayerName(const std::string layer_name)
        {
            auto search = name_to_layer_id_glb_.find(layer_name);
            if (search == name_to_layer_id_glb_.end())
            {
                name_to_layer_id_glb_.insert(std::pair<std::string, S32>(layer_name, getLayerID()));
            }
            else
            {
                std::cerr << "Layer name " << layer_name << " has been existed " << std::endl;
                std::cerr << "Exist ID " << search->second << " this ID " << getLayerID() << std::endl;
                Abort(-1);
            }
        }
        void Update(const F64 time)
        {
            queue_.PopSpk(time, dst_dinfo_);
            queue_.PushSpk(time, neuron_, dst_dinfo_);
            queue_.SyncAll(time, dst_dinfo_);
            /* if (Comm::getRank() == 0)
                queue_.dump(time, dst_dinfo_); */
        }
        void initRMA()
        {
            rma_queue_.initialize();
        }
        void freeRMA()
        {
            rma_queue_.free();
        }
        void initWorld()
        {
            const S32 n_loc = neuron_.getNumberOfParticleLocal();
            std::vector<Tspk> spk_loc;
            spk_loc.reserve(n_loc);
            for (S32 i = 0; i < n_loc; i++)
                spk_loc.push_back(Tspk(neuron_[i]));
            Comm::allGatherVAll(spk_loc, spk_loc.size(), spk_tot_);
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
        void CalcDynamics(Tfunc_dyn pfunc_dyn)
        {
            const F64 time_offset = GetWtime();
            pfunc_dyn(neuron_);
            time_profile.calc_dynamics += GetWtime() - time_offset;
        }
    };
    template <typename Tntyp>
    class Layer
    {
    public:
        typedef LayerInfo<typename Tntyp::Neuron,
                          typename Tntyp::Spike>
            Default;
    };
}