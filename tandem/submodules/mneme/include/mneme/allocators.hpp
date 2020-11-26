#ifndef MNEME_ALLOCATORS_H
#define MNEME_ALLOCATORS_H

#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>

namespace mneme {

struct StandardAllocatorBase {};
template <typename T>
struct StandardAllocator : public StandardAllocatorBase, public std::allocator<T> {};

struct AlignedAllocatorBase {};
template <class T, std::size_t Alignment> struct AlignedAllocator : public AlignedAllocatorBase {
    using value_type = T;
    constexpr static std::size_t alignment = Alignment;

    AlignedAllocator() = default;
    template <class U, std::size_t OtherAlignment>
    constexpr explicit AlignedAllocator(const AlignedAllocator<U, OtherAlignment>&) noexcept {}

    template <class U> struct rebind { typedef AlignedAllocator<U, Alignment> other; };

    [[nodiscard]] T* allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::bad_alloc();
        // Note: The alignment has to satisfy the following requirements
        // See also: https://www.man7.org/linux/man-pages/man3/posix_memalign.3.html
        constexpr bool isPowerOfTwo = (Alignment > 0 && ((Alignment & (Alignment - 1)) == 0));
        static_assert(isPowerOfTwo, "Alignment has to be a power of two");
        constexpr bool isFactorOfPtrSize = Alignment % sizeof(void*) == 0;
        static_assert(isFactorOfPtrSize, "Alignment has to be a constant of void ptr size");
        std::size_t size = n * sizeof(T);
        // size must be a multiple of Alignment
        // size := ceil(size/Alignment) * Alignment
        size = (size > 0U) ? (1U + (size - 1U) / Alignment) * Alignment : 0U;

        const auto ptr = reinterpret_cast<T*>(std::aligned_alloc(Alignment, size));
        auto isError = ptr == nullptr;
        if (isError) {
            throw std::bad_alloc();
        }

        return ptr;
    }
    void deallocate(T* ptr, std::size_t) noexcept { std::free(ptr); }
};

template <class T, std::size_t Alignment, class U, std::size_t OtherAlignment>
bool operator==(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, OtherAlignment>&) {
    return true;
}
template <class T, std::size_t Alignment, class U, std::size_t OtherAlignment>
bool operator!=(const AlignedAllocator<T, Alignment>&, const AlignedAllocator<U, OtherAlignment>&) {
    return false;
}

namespace detail {
template <typename, typename = void> inline constexpr bool hasAllocatorDefined = false;

template <typename T>
inline constexpr bool hasAllocatorDefined<T, std::void_t<decltype(sizeof(typename T::allocator))>> =
    true;
} // namespace detail

template <typename T, typename Default = StandardAllocator<typename T::type>>
struct AllocatorGetter {
    constexpr static auto makeAllocator() {
        if constexpr (detail::hasAllocatorDefined<T>) {
            return typename T::allocator();
        } else {
            return Default();
        }
    }
    using type = decltype(makeAllocator());
};

template <typename... List> struct AllocatorInfo;

template <typename Head, typename... Tail> struct AllocatorInfo<Head, Tail...> {

    template <typename Allocator> static constexpr bool allSameAllocatorAs() {
        using own_t = typename AllocatorGetter<Head>::type;
        return std::is_base_of_v<Allocator, own_t> &&
               AllocatorInfo<Tail...>::template allSameAllocatorAs<Allocator>();
    }

    static constexpr bool allSameAllocator() {
        using own_t = typename AllocatorGetter<Head>::type;
        static_assert(std::is_base_of_v<AlignedAllocatorBase, own_t> ||
                          std::is_base_of_v<StandardAllocatorBase, own_t>,
                      "AllocatorInfo::allSameAllocator is only defined for Aligned and Standard "
                      "allocator currently");
        if constexpr (std::is_base_of_v<AlignedAllocatorBase, own_t>) {
            return allSameAllocatorAs<AlignedAllocatorBase>();
        } else if constexpr (std::is_base_of_v<StandardAllocatorBase, own_t>) {
            return allSameAllocatorAs<StandardAllocatorBase>();
        } else {
            return allSameAllocatorAs<own_t>();
        }
    }

    static constexpr std::size_t getMaxAlignment() {
        using own_t = typename AllocatorGetter<Head>::type;
        static_assert(std::is_base_of_v<AlignedAllocatorBase, own_t>,
                      "Maximum alignment is only defined if all allocators are AlignedAllocator");
        return std::max(AllocatorInfo<Tail...>::getMaxAlignment(), own_t::alignment);
    }
};

template <> struct AllocatorInfo<> {
    template <typename Allocator> static constexpr auto allSameAllocatorAs() { return true; }

    static constexpr bool allSameAllocator() { return true; }
    static constexpr std::size_t getMaxAlignment() { return 0; }
};

template <typename... Args> constexpr bool allSameAllocator() {
    return AllocatorInfo<Args...>::allSameAllocator();
}
template <typename... Args> constexpr size_t getMaxAlignment() {
    constexpr auto maxAlignment = AllocatorInfo<Args...>::getMaxAlignment();
    return maxAlignment;
}

} // namespace mneme

#endif // MNEME_ALLOCATORS_H
