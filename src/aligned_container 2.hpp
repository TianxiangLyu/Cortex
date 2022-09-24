#include <vector>
#include <deque>
namespace Cortex
{
#define Alignment 1
#ifdef BOOST_ALIGN
    template <class T>
    using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, Alignment>>;
    template <class T>
    using aligned_deque = std::deque<T, boost::alignment::aligned_allocator<T, Alignment>>;
#else
    template <class T>
    using aligned_vector = std::vector<T>;
    template <class T>
    using aligned_deque = std::deque<T>;
#endif
}
