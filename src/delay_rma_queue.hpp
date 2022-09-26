#include <aligned_container.hpp>
#include <mpi.h>
namespace Cortex
{
    class RankInfo
    {
    public:
        S32 offset = -1;
        S32 num = -1;
    };
    template <class Tepj>
    class RmaRankRecv
    {
    public:
        std::vector<Tepj> recv_;
        // RmaRankRecv(const RankInfo &rank_info)
    };
    template <class Tepj>
    class RmaStepRecv
    {
    public:
        const S32 n_proc_;
        const std::vector<RankInfo> &rank_info_;
        const std::vector<Tepj> &spk_send_;
        std::vector<RmaRankRecv<Tepj>> rank_recv_;
        RmaStepRecv(const S32 n_proc,
                    const std::vector<RankInfo> &rank_info,
                    const std::vector<Tepj> &spk_send)
            : n_proc_(n_proc),
              rank_info_(rank_info),
              spk_send_(spk_send){};
    };
    template <class Tepj>
    class DelayRmaQueue
    {
    private:
        const S32 n_proc_;
        MPI_Win info_win_;
        MPI_Win send_win_;
        S32 step_tot_; // default
        S32 estimate_; // default
        DstDomainInfo &dst_dinfo_;
        F64 min_delay_;
        F64 max_delay_;
        std::vector<F64> delay_;

        std::vector<RankInfo> rank_info_; //[step][proc]
        S32 offset_ = 0;                  // for spk_send_;
        std::vector<Tepj> spk_send_;
        RmaStepRecv<Tepj> spk_recv_;

    public:
        DelayRmaQueue(DstDomainInfo &dst_dinfo, const S32 step_tot = 5000, const S32 estimate = 300)
            : n_proc_(dst_dinfo.getCommInfo().getNumberOfProc()),
              step_tot_(step_tot),
              estimate_(estimate),
              dst_dinfo_(dst_dinfo),
              rank_info_(step_tot * n_proc_),
              spk_send_(step_tot * estimate),
              spk_recv_(n_proc_, rank_info_, spk_send_)
        {
            for (S32 i = 0; i < step_tot * n_proc_; i++)
                assert(rank_info_[i].num == -1);
        };
        void free()
        {
            if (dst_dinfo_.getCommInfo().isNotCommNull())
            {
                MPI_Win_fence(0, info_win_);
                MPI_Win_fence(0, send_win_);
                MPI_Win_free(&info_win_);
                MPI_Win_free(&send_win_);
            }
        };
        void AddTarget(const S32 target_id, const F64 delay)
        {
            delay_.push_back(delay);
            min_delay_ = *std::min_element(delay_.begin(), delay_.end());
            max_delay_ = *std::max_element(delay_.begin(), delay_.end());
        }
        void initialize()
        {
            if (dst_dinfo_.getCommInfo().isNotCommNull())
            {
                MPI_Win_create(rank_info_.data(), rank_info_.size() * sizeof(RankInfo), sizeof(RankInfo),
                               MPI_INFO_NULL, dst_dinfo_.getCommInfo().getCommunicator(), &info_win_);
                MPI_Win_create(spk_send_.data(), spk_send_.size() * sizeof(Tepj), sizeof(Tepj),
                               MPI_INFO_NULL, dst_dinfo_.getCommInfo().getCommunicator(), &send_win_);
                MPI_Win_fence(0, info_win_);
                MPI_Win_fence(0, send_win_);
            }
        }
        template <class Tpsys>
        void PushSpk(const F64 time, Tpsys &psys, DstDomainInfo &dst_dinfo)
        {
        }
        void PopSpk(const F64 time, DstDomainInfo &dst_dinfo)
        {
        }
        typename std::deque<DelaySlot<Tepj>>::iterator find(const F64 time, const F64 delay)
        {
        }
        void dump(const F64 time, DstDomainInfo &dst_dinfo)
        {
        }
    };
}