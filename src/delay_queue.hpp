#include <aligned_container.hpp>
namespace Cortex
{
    template <class Tepj>
    class DelaySlot
    {
    public:
        bool sync = false;
        const F64 spk_time_;
        std::vector<Tepj> epj_send_;
        std::vector<Tepj> epj_recv_;
        template <class Tpsys>
        DelaySlot(const F64 time, Tpsys &psys)
            : spk_time_(time)
        {
            S32 n_epj_spk = 0;
            const S32 n_epj_tot = psys.getNumberOfParticleLocal();
            for (S32 i = 0; i < n_epj_tot; i++)
                if (psys[i].spike)
                    epj_send_.push_back(Tepj(psys[i]));
        };
        void dump()
        {
            std::cout << "Time " << spk_time_ << " Send " << epj_send_.size() << " Recv " << epj_recv_.size() << " Sync " << sync << std::endl;
        }
        void Sync(DstDomainInfo &dst_dinfo)
        {
            if (!sync)
            {
                dst_dinfo.getCommInfo().allGatherVAll(epj_send_, epj_send_.size(), epj_recv_);
                sync = true;
            }
        }
    };
    template <class Tepj>
    class DelayQueue
    {
    private:
        F64 min_delay_;
        F64 max_delay_;
        std::vector<F64> delay_;
        std::vector<S32> target_id_;
        std::deque<DelaySlot<Tepj>> queue_;

    public:
        DelayQueue(){};
        void AddTarget(const S32 target_id, const F64 delay)
        {
            delay_.push_back(delay);
            target_id_.push_back(target_id);
            min_delay_ = *std::min_element(delay_.begin(), delay_.end());
            max_delay_ = *std::max_element(delay_.begin(), delay_.end());
        }
        template <class Tpsys>
        void PushSpk(const F64 time, Tpsys &psys, DstDomainInfo &dst_dinfo)
        {
            if (dst_dinfo.getCommInfo().isNotCommNull())
                queue_.push_back(DelaySlot<Tepj>(time, psys));
        }
        void PopSpk(const F64 time, DstDomainInfo &dst_dinfo)
        {
            if (dst_dinfo.getCommInfo().isNotCommNull())
                while (!queue_.empty())
                {
                    if (queue_.front().spk_time_ + max_delay_ <= time - 1e-6)
                        queue_.pop_front();
                    else
                        break;
                }
        }
        typename std::deque<DelaySlot<Tepj>>::iterator find(const F64 time, const F64 delay)
        {
            typename std::deque<DelaySlot<Tepj>>::iterator it = queue_.begin();
            while (it != queue_.end() && it->spk_time_ + delay <= time - 1e-6)
                it++;
            return it;
        } // queue_.end() could be returned
        typename std::deque<DelaySlot<Tepj>>::reverse_iterator rfind(const F64 time, const F64 delay)
        {
            typename std::deque<DelaySlot<Tepj>>::reverse_iterator it = queue_.rbegin();
            while (it != queue_.rend() && it->spk_time_ + delay >= time + 1e-6)
                it++;
            return it;
        }
        typename std::deque<DelaySlot<Tepj>>::iterator begin() { return queue_.begin(); }
        typename std::deque<DelaySlot<Tepj>>::iterator end() { return queue_.end(); }
        typename std::deque<DelaySlot<Tepj>>::reverse_iterator rbegin() { return queue_.rbegin(); }
        typename std::deque<DelaySlot<Tepj>>::reverse_iterator rend() { return queue_.rend(); }
        void SyncAll(const F64 time, DstDomainInfo &dst_dinfo)
        {
            if (dst_dinfo.getCommInfo().isNotCommNull() && !queue_.empty())
                if (find(time, min_delay_) != queue_.end())
                    if (!find(time, min_delay_)->sync)
                        for (typename std::deque<DelaySlot<Tepj>>::iterator it = find(time, min_delay_); it != queue_.end(); it++)
                            it->Sync(dst_dinfo);
        }
        void dump(const F64 time, DstDomainInfo &dst_dinfo)
        {
            if (dst_dinfo.getCommInfo().isNotCommNull())
            {
                std::cout << std::endl;
                // std::cout << "Time " << time <<" find min "<<find(time, min_delay_)->spk_time_ <<" find max "<< find(time, max_delay_)->spk_time_ <<std::endl;
                std::cout << "Time " << time << std::endl;
                std::cout << std::endl;
                if (rfind(time, min_delay_) != queue_.rend())
                    std::cout << "rfind min " << rfind(time, min_delay_)->spk_time_ << std::endl;
                else
                    std::cout << "min delay no correspond slot " << std::endl;
                std::cout << std::endl;
                if (rfind(time, max_delay_) != queue_.rend())
                    std::cout << "rfind max " << rfind(time, max_delay_)->spk_time_ << std::endl;
                else
                    std::cout << "max delay no correspond slot " << std::endl;
                std::cout << std::endl;
                for (typename std::deque<DelaySlot<Tepj>>::iterator it = queue_.begin(); it != queue_.end(); it++)
                    it->dump();
                if (!find(time, min_delay_)->sync)
                    Abort(-1);
                // find(time, min_delay_)->sync;
            }
        }
    };
}