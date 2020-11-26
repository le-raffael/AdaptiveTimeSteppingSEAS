#ifndef MNEME_SPAN_H_
#define MNEME_SPAN_H_

#include <cstddef>
#include <limits>

#include "iterator.hpp"

namespace mneme {

inline constexpr std::size_t dynamic_extent = std::numeric_limits<std::size_t>::max();

template <typename T, std::size_t Extent = dynamic_extent> class span {
public:
    using iterator = Iterator<span<T, Extent>>;
    using const_iterator = Iterator<const span<T, Extent>>;

    span(T* base, std::size_t) : base(base) {}

    T& operator[](std::size_t idx) { return base[idx]; }
    T& operator[](std::size_t idx) const { return base[idx]; }
    T* data() { return base; }
    T* data() const { return base; }

    std::size_t size() const { return Extent; }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size()); }

    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, size()); }

private:
    T* base = nullptr;
};

template <typename T> class span<T, dynamic_extent> {
public:
    using iterator = Iterator<span<T, dynamic_extent>>;
    using const_iterator = Iterator<const span<T, dynamic_extent>>;

    span(T* base, std::size_t extent) : base(base), extent(extent) {}

    T& operator[](std::size_t idx) { return base[idx]; }
    T& operator[](std::size_t idx) const { return base[idx]; }
    T* data() { return base; }
    T* data() const { return base; }

    std::size_t size() const { return extent; }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size()); }

    const_iterator begin() const { return const_iterator(this, 0); }
    const_iterator end() const { return const_iterator(this, size()); }

private:
    T* base = nullptr;
    std::size_t extent;
};

} // namespace mneme

#endif // MNEME_SPAN_H_
