#include <vector>
#include <cortex_defs.hpp>
#include <population.hpp>
namespace Cortex
{
    enum RECORD_SELECT_MODE
    {
        SERIAL,
        RANDOM
    };
    class Recorder
    {
    private:
        std::vector<S32> adr_for_record_;

    public:
        template <class Tneu, class Tspk>
        Recorder(PopulationInfo<Tneu, Tspk> &src,
                 const S32 record_num,
                 const RECORD_SELECT_MODE mode = RECORD_SELECT_MODE::SERIAL){

        };
    };

}