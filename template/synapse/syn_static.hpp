#include <cortex.hpp>
#include <mutex>
#include <vector>
namespace CX = Cortex;
template <class params>
class syn_static
{
public:
    constexpr static const CX::F64 delay = params::delay;
    struct LinkInfo
    {
        struct Link
        {
            CX::S32 target;
            CX::F32 weight;
            Link(){};
            Link(const CX::S32 _target, const CX::F32 _weight)
                : target(_target),
                  weight(_weight){};
        };
        CX::S32 n_link;
        Link *info;
        void init(const CX::S32 num)
        {
            this->n_link = num;
            info = new CX::S32[num];
        }
        void setLink(const CX::S32 id, const CX::S32 target) { this->info[id].target = target; }
        void setWeight(const CX::S32 id, const CX::F64 value) { this->info[id].weight = value; }
        ~LinkInfo()
        {
            delete[] info;
        }
    };
    struct Post
    {
        CX::S32 id;
        CX::F64vec pos;
        CX::S32 randSeed;
        CX::F32 Rsearch;
        CX::F32 peak;
        CX::F32 t_lastSpike;
        CX::S32 n_link;
        CX::F64 &input;
        template <class Tfp>
        Post(Tfp &fp, CX::F64 &_input)
            : id(fp.id),
              pos(fp.pos),
              randSeed(fp.randSeed),
              Rsearch(fp.Rsearch),
              input(_input){};
        template <class Tep>
        void setFromEP(const Tep &ep)
        {
            this->id = ep.id;
            this->pos = ep.pos;
            this->randSeed = ep.randSeed;
            this->Rsearch = ep.Rsearch;
        } // no peak because peak might be different in different connection which is not a value in a layer
        inline void updateSpk(const CX::F64 time, const CX::F64 max_delay = delay)
        {
        }
    };
    struct Synapse
    {
        const CX::S32 id;
        LinkInfo &link;
        template <class Tep>
        Synapse(Tep ep, LinkInfo &_link)
            : id(ep.id),
              link(_link){};
    };
    class CalcInteraction
    {
    private:
        const CX::F64 weight;

    public:
        CalcInteraction(const CX::F64 _weight)
            : weight(_weight){};
        void operator()(Post *const ep_i, const CX::S32 begin_i, const CX::S32 end_i,
                        const Synapse *const ep_j, const CX::S32 Njp)
        {
            for (CX::S32 j = 0; j < Njp; j++)
            {
                const CX::S32 lo = LFsearch(ep_j[j].link.info, ep_j[j].link.n_link, begin_i);
                const CX::S32 hi = RHsearch(ep_j[j].link.info, ep_j[j].link.n_link, end_i);
                for (CX::S32 i = 0; i < ep_j[j].link.n_link; i++)
                {
                    const CX::S32 adr = ep_j[j].link.link[i];
                    // ep_i[adr].input += ep_j[j].link.weight[i];
                }
            }
        }
    };
};