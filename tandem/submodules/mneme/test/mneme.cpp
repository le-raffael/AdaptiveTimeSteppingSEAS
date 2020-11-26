#include "mneme/plan.hpp"
#include "mneme/storage.hpp"
#include "mneme/view.hpp"

#include "doctest.h"

#include <array>
#include <functional>
#include <memory>
#include <optional>
using namespace mneme;

struct ElasticMaterial {
    double rho;
    double mu;
    double lambda;
};

struct dofsAligned {
    using type = double;
    using allocator = StandardAllocator<type>;
};
struct material {
    using type = ElasticMaterial;
    using allocator = StandardAllocator<type>;
};
struct dofs {
    using type = double;
};
struct bc {
    using type = std::array<int, 4>;
    using allocator = StandardAllocator<type>;
};

struct Ghost : public Layer {};
struct Interior : public Layer {};
struct Copy : public Layer {};

TEST_CASE("Data structure works") {
    int NghostP1 = 5;
    int NinteriorP1 = 100;
    int NinteriorP2 = 50;
    int N = NghostP1 + NinteriorP1 + NinteriorP2;

    Plan localPlan(N);
    for (int i = NghostP1; i < NghostP1 + NinteriorP1 + NinteriorP2; ++i) {
        localPlan.setDof(i, 1); // Store one object per i
    }
    auto localLayout = localPlan.getLayout();

    Plan dofsPlan(N);
    int idx = 0;
    for (int i = 0; i < NghostP1 + NinteriorP1; ++i) {
        dofsPlan.setDof(idx++, 4); // Store P1 element per i
    }
    for (int i = 0; i < NinteriorP2; ++i) {
        dofsPlan.setDof(idx++, 10); // Store P2 element per i
    }
    auto dofsLayout = dofsPlan.getLayout();

    using aos_t = MultiStorage<DataLayout::AoS, material, bc>;
    using soa_t = MultiStorage<DataLayout::SoA, material, bc>;
    auto localAoS = std::make_shared<aos_t>(localLayout.back());
    auto localSoA = std::make_shared<soa_t>(localLayout.back());
    auto testMaterial = [](auto&& X) {
        for (int i = 0; i < static_cast<int>(X.size()); ++i) {
            X[i].template get<material>() = ElasticMaterial{1.0 * i, 1.0 * i, 2.0 * i};
        }
        int i = 0;
        for (auto&& x : X) {
            REQUIRE(x.template get<material>().lambda == 2.0 * i++);
        }
    };

    SUBCASE("MultiStorage AoS works") { testMaterial(*localAoS); }

    SUBCASE("MultiStorage SoA works") { testMaterial(*localSoA); }

    SUBCASE("DenseView AoS works") {
        DenseView<aos_t> localView(localLayout, localAoS, NghostP1, N);
        testMaterial(localView);
    }

    SUBCASE("DenseView SoA works") {
        DenseView<soa_t> localView(localLayout, localSoA, NghostP1, N);
        testMaterial(localView);
    }

    SUBCASE("DenseView AoS works") {
        DenseView<aos_t> localView(localLayout, localAoS, NghostP1, N);
        testMaterial(localView);
    }

    SUBCASE("SingleStorage works") {
        using dofs_storage_t = SingleStorage<dofsAligned>;
        auto dofsC = std::make_shared<dofs_storage_t>(dofsLayout.back());

        StridedView<dofs_storage_t, 4U> dofsV(dofsLayout, dofsC, 0, NghostP1 + NinteriorP1);
        int k = 0;
        int l = 0;
        for (auto&& v : dofsV) {
            l = 0;
            for (auto&& vv : v) {
                vv = k + 4 * l++;
            }
            ++k;
        }
        for (int j = 0; j < NghostP1 + NinteriorP1; ++j) {
            REQUIRE((*dofsC)[j] == j / 4 + 4 * (j % 4));
        }
    }
}

TEST_CASE("Layered Plans") {
    constexpr std::size_t numInterior = 100;
    constexpr std::size_t numCopy = 30;
    constexpr std::size_t numGhost = 20;

    constexpr std::size_t numMaterial = 1;
    constexpr auto numMaterialFunc = [numMaterial](auto) { return numMaterial; };
    const auto localPlan = LayeredPlan()
                               .withDofs<Interior>(numInterior, numMaterialFunc)
                               .withDofs<Copy>(numInterior, numMaterialFunc)
                               .withDofs<Ghost>(numInterior, numMaterialFunc);
    const auto& localLayout = localPlan.getLayout();
    CHECK(std::is_same_v<decltype(localPlan), const LayeredPlan<Interior, Copy, Ghost>>);

    SUBCASE("Layout is cached") {
        const auto& localLayout2 = localPlan.getLayout();
        CHECK(&localLayout == &localLayout2);
        CHECK(localLayout.data() == localLayout2.data());
    }
    SUBCASE("Caching of layout is invalidated") {
        const auto tmpPlan = LayeredPlan()
                                 .withDofs<Interior>(numInterior, numMaterialFunc)
                                 .withDofs<Copy>(numInterior, numMaterialFunc);
        [[maybe_unused]] const auto& tmpLayout = tmpPlan.getLayout();
        const auto localPlan2 = tmpPlan.withDofs<Ghost>(numInterior, numMaterialFunc);
        const auto& localLayout2 = localPlan2.getLayout();
        CHECK(localLayout.size() == localLayout2.size());
        for (std::size_t i = 0; i < localLayout.size(); ++i) {
            CHECK(localLayout[i] == localLayout2[i]);
        }
    }

    constexpr std::size_t numDofsInterior = 10;
    constexpr std::size_t numDofsCopy = 7;
    constexpr std::size_t numDofsGhost = 4;
    constexpr auto dofsInterior = [numDofsInterior](auto) { return numDofsInterior; };
    constexpr auto dofsCopy = [numDofsCopy](auto) { return numDofsCopy; };
    constexpr auto dofsGhost = [numDofsGhost](auto) { return numDofsGhost; };
    const auto dofsPlan = LayeredPlan()
                              .withDofs<Interior>(numInterior, dofsInterior)
                              .withDofs<Copy>(numCopy, dofsCopy)
                              .withDofs<Ghost>(numGhost, dofsGhost);
    CHECK(std::is_same_v<decltype(dofsPlan), const LayeredPlan<Interior, Copy, Ghost>>);
    const auto dofsLayout = dofsPlan.getLayout();

    CHECK(dofsLayout.size() == numInterior + numCopy + numGhost);
    std::size_t curOffset = 0;
    std::size_t i = 0;
    for (; i < numInterior; ++i) {
        CHECK(dofsLayout[i] == curOffset);
        curOffset += numDofsInterior;
    }
    for (; i < numCopy; ++i) {
        CHECK(dofsLayout[i] == curOffset);
        curOffset += numDofsCopy;
    }
    for (; i < numGhost; ++i) {
        CHECK(dofsLayout[i] == curOffset);
        curOffset += numDofsGhost;
    }

    SUBCASE("MultiStorage works") {
        using local_storage_t = MultiStorage<DataLayout::SoA, material, bc>;
        auto localC = std::make_shared<local_storage_t>(localLayout.back());

        auto materialViewFactory = createViewFactory().withPlan(localPlan).withStorage(localC);

        auto localViewCopy = materialViewFactory.createDenseView<Copy>();

        for (int i = 0; i < static_cast<int>(localViewCopy.size()); ++i) {
            localViewCopy[i].get<material>() = ElasticMaterial{1.0 * i, 1.0 * i, 2.0 * i};
        }
        int i = 0;
        for (auto&& v : localViewCopy) {
            CHECK(v.get<material>().lambda == 2.0 * i++);
        }
    }
    SUBCASE("SingleStorage works") {
        using dofs_storage_t = SingleStorage<dofs>;
        auto dofsC = std::make_shared<dofs_storage_t>(dofsLayout.back());

        auto dofsViewFactory = createViewFactory().withPlan(dofsPlan).withStorage(dofsC);
        auto dofsViewInterior =
            dofsViewFactory.withStride<numDofsInterior>().createStridedView<Interior>();
        int k = 0;
        int l = 0;
        for (auto&& v : dofsViewInterior) {
            l = 0;
            for (auto&& vv : v) {
                vv = k + 10 * l++;
            }
            ++k;
        }
        for (std::size_t j = 0; j < numInterior; ++j) {
            CHECK((*dofsC)[j] == j / 10 + 10 * (j % 10));
        }
    }
    SUBCASE("Combined plan") {
        auto plans = std::vector{localPlan, localPlan};
        auto combinedPlan = CombinedLayeredPlan(plans);
        using local_storage_t = MultiStorage<DataLayout::SoA, material, bc>;
        auto combinedLayout = combinedPlan.getLayout();
        auto localC = std::make_shared<local_storage_t>(combinedLayout.back());
        auto materialViewFactory = createViewFactory().withPlan(combinedPlan).withStorage(localC);
        auto localViewCopyCluster0 = materialViewFactory.withClusterId(0).createDenseView<Copy>();
        auto localViewCopyCluster1 = materialViewFactory.withClusterId(1).createDenseView<Copy>();
        auto localViewInteriorCluster0 =
            materialViewFactory.withClusterId(0).createDenseView<Interior>();
        auto localViewInteriorCluster1 =
            materialViewFactory.withClusterId(1).createDenseView<Interior>();
        for (std::size_t i = 0; i < localViewInteriorCluster0.size(); ++i) {
            localViewInteriorCluster0[i].get<material>() =
                ElasticMaterial{1.0 * i, 1.0 * i, 2.0 * i};
        }
        for (std::size_t i = 0; i < localViewInteriorCluster1.size(); ++i) {
            localViewInteriorCluster1[i].get<material>() =
                ElasticMaterial{2.0 * i, 2.0 * i, 4.0 * i};
        }
        for (std::size_t i = 0; i < localViewCopyCluster0.size(); ++i) {
            localViewCopyCluster0[i].get<material>() = ElasticMaterial{3.0 * i, 3.0 * i, 6.0 * i};
        }
        for (std::size_t i = 0; i < localViewCopyCluster1.size(); ++i) {
            localViewCopyCluster1[i].get<material>() = ElasticMaterial{4.0 * i, 4.0 * i, 8.0 * i};
        }
        int i = 0;
        for (auto&& v : localViewInteriorCluster0) {
            CHECK(v.get<material>().lambda == 2.0 * i++);
        }
        i = 0;
        for (auto&& v : localViewInteriorCluster1) {
            CHECK(v.get<material>().lambda == 4.0 * i++);
        }
        i = 0;
        for (auto&& v : localViewCopyCluster0) {
            CHECK(v.get<material>().lambda == 6.0 * i++);
        }
        i = 0;
        for (auto&& v : localViewCopyCluster1) {
            CHECK(v.get<material>().lambda == 8.0 * i++);
        }
    }
}
