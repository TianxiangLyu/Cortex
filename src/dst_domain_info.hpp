#ifdef CORTEX_MPI_PARALLEL
#include <mpi.h>
#endif
#include <unordered_map>
#include <cortex.hpp>
namespace Cortex
{
    class DstPosInfo
    {
    private:
        S32 target_id_;
        F64 delay_;
        CommInfo comm_info_;
        std::vector<F64ort> dst_pos_domain_;

    public:
        DstPosInfo(const S32 target, const F64 delay, DomainInfo &d)
            : target_id_(target),
              delay_(delay)
        {
            comm_info_ = d.getCommInfo();
            const S32 n_proc = comm_info_.getNumberOfProc();
            dst_pos_domain_.resize(n_proc);
            for (S32 i = 0; i < n_proc; i++)
                dst_pos_domain_[i] = d.getPosDomain(i);
        };
        F64ort & operator[] (const int i) { return dst_pos_domain_[i]; }
        S32 getTargetID() const {return target_id_;}
        F64 getDelay() const { return delay_; }
        CommInfo getCommInfo() const { return comm_info_; }
        void dump(const S32 id)
        {
            std::cout << "Dst id " << id << std::endl;
            std::cout << std::endl;
            for (S32 i = 0; i < comm_info_.getNumberOfProc(); i++)
                std::cout<<comm_info_.transRank(i, Comm::getCommInfo())<<" ";
            std::cout << std::endl;
            std::cout << std::endl;
            for (S32 i = 0; i < dst_pos_domain_.size(); i++)
                std::cout << dst_pos_domain_[i] << std::endl;
            std::cout << std::endl;
        }
    };
    class DstDomainInfo
    {
    private:
        CommInfo src_comm_info_;
        CommInfo comm_info_;
        std::vector<F64ort> src_pos_domain_;
        std::vector<DstPosInfo> dst_pos_info_;
        BOUNDARY_CONDITION boundary_condition_;
        bool periodic_axis_[DIMENSION_LIMIT]; // for src only
        F64ort pos_root_domain_;

    public:
        DstDomainInfo()
            : comm_info_(){};
        DstDomainInfo(MPI_Group group)
            : comm_info_(group){};
        S32 getDstNum() { return dst_pos_info_.size(); }
        DstPosInfo & operator[] (const int i) { return dst_pos_info_[i]; }
        CommInfo getCommInfo() const { return comm_info_; }
        CommInfo getSrcCommInfo() const { return src_comm_info_; }
        const F64ort getPosRootDomain() const { return pos_root_domain_; }
        const F64vec getLenRootDomain() const { return pos_root_domain_.getFullLength(); }
        S32 getBoundaryCondition() const { return boundary_condition_; }
        void setBoundaryCondition(enum BOUNDARY_CONDITION bc)
        {
            boundary_condition_ = bc;
            if (DIMENSION == 2 &&
                (bc == BOUNDARY_CONDITION_PERIODIC_XYZ ||
                 bc == BOUNDARY_CONDITION_PERIODIC_XZ ||
                 bc == BOUNDARY_CONDITION_PERIODIC_YZ ||
                 bc == BOUNDARY_CONDITION_PERIODIC_Z))
            {
                throw "CX_ERROR: in setBoundaryCondition(enum BOUNDARY_CONDITION) \n boundary condition is incompatible with DIMENSION";
            }
            if (bc == BOUNDARY_CONDITION_OPEN)
                periodic_axis_[0] = periodic_axis_[1] = periodic_axis_[2] = false;
            else if (bc == BOUNDARY_CONDITION_PERIODIC_X)
            {
                periodic_axis_[0] = true;
                periodic_axis_[1] = periodic_axis_[2] = false;
            }
            else if (bc == BOUNDARY_CONDITION_PERIODIC_Y)
            {
                periodic_axis_[1] = true;
                periodic_axis_[0] = periodic_axis_[2] = false;
            }
            else if (bc == BOUNDARY_CONDITION_PERIODIC_Z)
            {
                periodic_axis_[2] = true;
                periodic_axis_[0] = periodic_axis_[1] = false;
            }
            else if (bc == BOUNDARY_CONDITION_PERIODIC_XY)
            {
                periodic_axis_[0] = periodic_axis_[1] = true;
                periodic_axis_[2] = false;
            }
            else if (bc == BOUNDARY_CONDITION_PERIODIC_XZ)
            {
                periodic_axis_[0] = periodic_axis_[2] = true;
                periodic_axis_[1] = false;
            }
            else if (bc == BOUNDARY_CONDITION_PERIODIC_YZ)
            {
                periodic_axis_[1] = periodic_axis_[2] = true;
                periodic_axis_[0] = false;
            }
            else if (bc == BOUNDARY_CONDITION_PERIODIC_XYZ)
                periodic_axis_[0] = periodic_axis_[1] = periodic_axis_[2] = true;
        }
        void getPeriodicAxis(bool pa[]) const
        {
            for (S32 i = 0; i < DIMENSION; i++)
                pa[i] = periodic_axis_[i];
        }
        void setSrcInfo(DomainInfo &d)
        {
            src_comm_info_ = d.getCommInfo();
            comm_info_ = d.getCommInfo();
            pos_root_domain_ = d.getPosRootDomain();
            const S32 n_proc = comm_info_.getNumberOfProc();
            setBoundaryCondition((enum BOUNDARY_CONDITION)d.getBoundaryCondition());
            src_pos_domain_.resize(n_proc);
            for (S32 i = 0; i < n_proc; i++)
                src_pos_domain_[i] = d.getPosDomain(i);
        }
        void addDstInfo(const S32 target, const F64 delay, DomainInfo &d)
        {
            comm_info_.Union(d.getCommInfo());
            dst_pos_info_.push_back(DstPosInfo(target, delay, d));
        }
        void dump()
        {
            std::cout << "Src Rank " << std::endl;
            std::cout << std::endl;
            for (S32 i = 0; i < src_comm_info_.getNumberOfProc(); i++)
                std::cout << src_comm_info_.transRank(i, Comm::getCommInfo()) << " ";
            std::cout << std::endl;
            std::cout << std::endl;

            std::cout << "Dst Rank " << std::endl;
            std::cout << std::endl;
            for (S32 i = 0; i < comm_info_.getNumberOfProc(); i++)
                std::cout << comm_info_.transRank(i, Comm::getCommInfo()) << " ";
            std::cout << std::endl;
            std::cout << std::endl;

            std::cout << "Src " << std::endl;
            std::cout << std::endl;
            for (S32 i = 0; i < src_comm_info_.getNumberOfProc(); i++)
                std::cout << src_pos_domain_[i] << std::endl;
            std::cout << std::endl;
            for (S32 i = 0; i < dst_pos_info_.size(); i++)
                dst_pos_info_[i].dump(i);
        }
    };
}