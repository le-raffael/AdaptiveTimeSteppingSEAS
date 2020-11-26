#include "mneme/allocators.hpp"
#include "doctest.h"
#include "mneme/plan.hpp"
#include "mneme/storage.hpp"
#include "mneme/view.hpp"
#include <cstdint>
#include <vector>

using namespace mneme;

template <typename T> void checkPointerAlignment(T* ptr, std::size_t alignment) {
    CHECK(reinterpret_cast<uintptr_t>(ptr) % alignment == 0);
}

TEST_CASE("Aligned allocator works") {
    // Note: Alignment has to be carefully chosen here,
    // as often pointers are aligned even with default allocator.
    // E.g. the test would pass with alignment=32 even with malloc
    // on my machine.
    constexpr static auto alignment = 1024U;
    using value_t = int;
    using allocator_t = AlignedAllocator<value_t, alignment>;
    using allocator_traits_t = std::allocator_traits<allocator_t>;
    auto allocator = allocator_t();
    auto* mem = allocator_traits_t::allocate(allocator, 100);
    checkPointerAlignment(mem, alignment);
    allocator_traits_t::deallocate(allocator, mem, 100);

    SUBCASE("Aligned allocator can be used with std containers") {
        auto alignedVec = std::vector<value_t, allocator_t>(1000);
        checkPointerAlignment(alignedVec.data(), alignment);
    }
}

constexpr static std::size_t alignment = 2048 * 4;
struct dofsAligned {
    using type = double;
    using allocator = AlignedAllocator<type, alignment>;
};

struct dofsAlignedLarge {
    using type = double;
    using allocator = AlignedAllocator<type, alignment * 2>;
};

struct dofsUnaligned {
    using type = double;
};

TEST_CASE("Combining allocator types") {
    CHECK(allSameAllocator<dofsAligned, dofsAlignedLarge>());
    CHECK(!allSameAllocator<dofsAligned, dofsAlignedLarge, dofsUnaligned>());
    constexpr auto maxAlignment = getMaxAlignment<dofsAligned, dofsAlignedLarge>();
    CHECK(std::max(dofsAligned::allocator::alignment, dofsAlignedLarge::allocator::alignment) ==
          maxAlignment);
}

TEST_CASE("Storage works with aligned allocator") {
    constexpr std::size_t numElements = 10;
    constexpr std::size_t numDofs = 7;
    Plan dofsPlan(numElements);
    int idx = 0;
    for (std::size_t i = 0; i < numElements; ++i) {
        dofsPlan.setDof(idx++, numDofs);
    }
    auto dofsLayout = dofsPlan.getLayout();
    using dofs_storage_t = SingleStorage<dofsAligned>;
    auto dofsC = std::make_shared<dofs_storage_t>(dofsLayout.back());
    auto dofsV = StridedView<dofs_storage_t, numDofs>(dofsLayout, dofsC, 0, numElements);

    checkPointerAlignment(dofsV[0].data(), alignment);
}

TEST_CASE("Alignment works MultiStorage") {
    constexpr std::size_t numElements = 10;
    Plan dofsPlan(numElements);
    int idx = 0;
    for (std::size_t i = 0; i < numElements; ++i) {
        dofsPlan.setDof(idx++, 1);
    }
    auto dofsLayout = dofsPlan.getLayout();

    SUBCASE("SoA Multistorage is aligned") {
        using local_storage_soa_t = MultiStorage<DataLayout::SoA, dofsAligned, dofsAlignedLarge>;
        auto localStorageSoA = std::make_shared<local_storage_soa_t>(dofsLayout.back());
        auto localViewSoA =
            DenseView<local_storage_soa_t>(dofsLayout, localStorageSoA, 0, numElements);
        checkPointerAlignment(&localViewSoA[0].get<dofsAligned>(), alignment);
        checkPointerAlignment(&localViewSoA[0].get<dofsAlignedLarge>(), 2 * alignment);
    }
    SUBCASE("AoS Multistorage is aligned") {
        using local_storage_aos_t = MultiStorage<DataLayout::AoS, dofsAligned, dofsAlignedLarge>;
        auto localStorageAoS = std::make_shared<local_storage_aos_t>(dofsLayout.back());
        [[maybe_unused]] auto localViewAoS =
            DenseView<local_storage_aos_t>(dofsLayout, localStorageAoS, 0, numElements);

        // Only check compilation, no guarantees on tuple storage layout
    }
}

struct A {
    using type = double;
    using allocator = AlignedAllocator<type, alignment>;
};

struct B {
    using type = int;
};

TEST_CASE("AllocatorGetter constructs correct default values") {
    auto allocA = AllocatorGetter<A, StandardAllocator<A::type>>::makeAllocator();
    CHECK(std::is_base_of_v<AlignedAllocatorBase, decltype(allocA)>);
    auto allocBDefaultNonAligned = AllocatorGetter<B, StandardAllocator<B::type>>::makeAllocator();
    CHECK(std::is_base_of_v<StandardAllocatorBase, decltype(allocBDefaultNonAligned)>);
    auto allocBDefaultAligned =
        AllocatorGetter<B, AlignedAllocator<B::type, alignment>>::makeAllocator();
    CHECK(std::is_base_of_v<AlignedAllocatorBase, decltype(allocBDefaultAligned)>);
}

TEST_CASE("Default allocator compiles") {
    struct IdWithoutAllocator {
        using type = int;
    };
    auto testAoS = MultiStorage<DataLayout::AoS, IdWithoutAllocator>(5);
    auto testSoA = MultiStorage<DataLayout::SoA, IdWithoutAllocator>(5);
    auto testSingle = SingleStorage<IdWithoutAllocator>(5);
}
