#ifndef MNEME_PLAN_H_
#define MNEME_PLAN_H_

#include <cstddef>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "displacements.hpp"
#include "span.hpp"
#include "tagged_tuple.hpp"

namespace mneme {

class Plan {
public:
    using layout_t = Displacements<std::size_t>;
    explicit Plan(std::size_t numElements) : dofs(numElements, 0) {}

    void setDof(std::size_t elementNo, std::size_t dof) { dofs[elementNo] = dof; }

    void resize(std::size_t newSize) { dofs.resize(newSize); }
    [[nodiscard]] layout_t getLayout() const { return Displacements(dofs); }

private:
    std::vector<std::size_t> dofs;
};

class Layer {
public:
    constexpr Layer() = default;
    constexpr Layer(std::size_t numElements, std::size_t offset)
        : numElements(numElements), offset(offset) {}
    std::size_t numElements = 0;
    std::size_t offset = 0;
};

class LayeredPlanBase {};

/***
 * This class implements a way of creating plans with multiple layers.
 * You can use it like this:
 * constexpr auto dofsInterior = [numDofsInterior](auto) { return numDofsInterior; };
 *  constexpr auto dofsCopy = [numDofsCopy](auto) { return numDofsCopy; };
 *  constexpr auto dofsGhost = [numDofsGhost](auto) { return numDofsGhost; };
 *  const LayeredPlan<Interior, Copy, Ghost> dofsPlan = LayeredPlan()
 *                       .withDofs<Interior>(numInterior, dofsInterior)
 *                       .withDofs<Copy>(numCopy, dofsCopy)
 *                       .withDofs<Ghost>(numGhost, dofsGhost);
 *  dofsInterior, dofsCopy, dofsGhost are functions that return the number of dofs
 *  for each index/layer.
 *  Afterwards, you can use dofsPlan like every other plan.
 * @tparam Layers is a list of Layers that inherit from the type Layer.
 */
template <typename... Layers> class LayeredPlan : public LayeredPlanBase {
    template <typename... OtherLayers> friend class LayeredPlan;

public:
    using layout_t = Displacements<std::size_t>;
    LayeredPlan() : plan(0) {}

    LayeredPlan(std::size_t curOffset, std::size_t numElements, Plan plan,
                std::tuple<Layers...> layers)
        : curOffset(curOffset), numElements(numElements), plan(std::move(plan)), layers(layers),
          layout(std::nullopt) {}

    template <typename OtherPlanT>
    explicit LayeredPlan(OtherPlanT otherPlan) : plan(otherPlan.plan) {
        this->curOffset = otherPlan.curOffset;
        this->numElements = otherPlan.numElements;
        this->plan = otherPlan.plan;
        // Copy all entries of old tuple into new tuple.
        std::apply(
            [&](auto&&... args) {
                (((std::get<typename std::remove_reference_t<std::remove_const_t<decltype(args)>>>(
                      layers)) =
                      std::get<typename std::remove_reference_t<decltype(args)>>(otherPlan.layers)),
                 ...);
            },
            otherPlan.layers);
    }

    template <typename Layer, typename Func>
    LayeredPlan<Layers..., Layer> withDofs(std::size_t numElementsLayer, Func func) const {
        auto newPlan = LayeredPlan<Layers..., Layer>(*this);
        auto& newLayer = std::get<Layer>(newPlan.layers);
        newLayer.numElements = numElementsLayer;
        newLayer.offset = curOffset;
        const auto newCurOffset = curOffset + numElementsLayer;
        const auto newNumElements = numElements + numElementsLayer;
        newPlan.curOffset = newCurOffset;
        newPlan.numElements = newNumElements;

        newPlan.plan.resize(newNumElements);

        for (std::size_t i = 0; i < newLayer.numElements; ++i) {
            newPlan.plan.setDof(i + newLayer.offset, func(i));
        }
        return newPlan;
    }

    const layout_t& getLayout() const {
        if (layout == std::nullopt) {
            layout = plan.getLayout();
        }
        return *layout;
    }

    template <typename T> T getLayer() const { return std::get<T>(layers); }
    std::size_t getOffset() const { return curOffset; }
    size_t size() const { return numElements; };

private:
    std::tuple<Layers...> layers;
    std::size_t curOffset = 0;
    std::size_t numElements = 0;
    Plan plan;
    mutable std::optional<layout_t> layout;
};

class CombinedLayeredPlanBase {};

/**
 * This class allows you to combine multiple LayeredPlans into one.
 * You can use this to have multiple plans for subsets of your domain
 * but with one single Layout.
 * You can use it like this:
 * auto plans = std::vector{localPlan, localPlan};
 * auto combinedPlan = CombinedLayeredPlan(plans);
 * @tparam Layers is a list of Layers that inherit from the type Layer.
 */
template <typename... Layers> class CombinedLayeredPlan : public CombinedLayeredPlanBase {
public:
    using plan_t = LayeredPlan<Layers...>;
    using layout_t = Displacements<size_t>;

    explicit CombinedLayeredPlan(std::vector<plan_t> plans) : plans(plans), offsets(plans.size()) {
        std::size_t offset = 0;
        for (std::size_t i = 0; i < plans.size(); ++i) {
            const auto& plan = plans[i];
            offsets[i] = offset;
            offset += plan.size();
        }
    }

    [[nodiscard]] layout_t getLayout() const {
        const std::size_t totalNumberOfDofs =
            std::accumulate(plans.begin(), plans.end(), 0U,
                            [](auto count, auto& vec) { return count + vec.size(); });

        std::vector<size_t> combinedDofs(totalNumberOfDofs);
        std::size_t idx = 0;
        for (auto i = 0U; i < plans.size(); ++i) {
            const auto& plan = plans[i];
            const auto& curLayout = plan.getLayout();
            for (auto j = 0U; j < curLayout.size(); ++j) {
                combinedDofs[idx] = curLayout.count(j);
                ++idx;
            }
        }
        return {combinedDofs};
    }

    template <typename T> T getLayer(std::size_t clusterId) const {
        const auto& cluster = plans[clusterId];
        auto layer = cluster.template getLayer<T>();
        layer.offset += offsets[clusterId];
        return layer;
    }

private:
    std::vector<plan_t> plans;
    std::vector<std::size_t> offsets;
};
} // namespace mneme

#endif // MNEME_PLAN_H_
