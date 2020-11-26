#include "doctest.h"
#include <cstddef>
#include <utility>
#include <vector>

#include "mneme/displacements.hpp"

using mneme::Displacements;

TEST_CASE("testing displacements") {
    Displacements<int> displacements;
    std::vector<int> count{0, 4, 0, 0, 1, 0, 2, 0};

    SUBCASE("make displacements") {
        displacements.make(count);

        REQUIRE(displacements.size() == count.size());
        for (size_t p = 0; p < displacements.size(); ++p) {
            CHECK(count[p] == displacements.count(p));
        }
        CHECK(displacements[displacements.size()] == 7);
    }

    SUBCASE("empty displacements iterator") {
        std::size_t numIterations = 0;
        for (auto [p, i] : displacements) {
            CHECK(p == 0);
            CHECK(i == 0);
            ++numIterations;
        }
        CHECK(numIterations == 0);
    }

    SUBCASE("displacements iterator") {
        displacements.make(count);
        std::size_t numIterations = 0;
        std::vector<std::pair<int, int>> pairs;
        for (auto pi : displacements) {
            ++numIterations;
            pairs.emplace_back(pi);
        }
        CHECK(numIterations == 7);

        auto it = pairs.begin();
        for (size_t p = 0; p < count.size(); ++p) {
            for (int i = displacements[p]; i < displacements[p + 1]; ++i) {
                CHECK(it->first == p);
                CHECK(it->second == i);
                ++it;
            }
        }
    }
}
