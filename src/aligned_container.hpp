#include <vector>
#include <deque>
#include <array>
namespace Cortex
{
#define Alignment 64
#ifdef BOOST_ALIGN
    template <class T>
    using aligned_vector = std::vector<T, boost::alignment::aligned_allocator<T, Alignment>>;
    template <class T>
    using aligned_deque = std::deque<T, boost::alignment::aligned_allocator<T, Alignment>>;
    template <class T>
    using aligned_array = std::array<T, boost::alignment::aligned_allocator<T, Alignment>>;
#else
    template <class T>
    using aligned_vector = std::vector<T>;
    template <class T>
    using aligned_deque = std::deque<T>;
    template <class T, unsigned int cap>
    using aligned_array = std::array<T, cap>;
#endif
}
