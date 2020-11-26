#ifndef MNEME_VIEW_H_
#define MNEME_VIEW_H_

#include "iterator.hpp"
#include "plan.hpp"
#include "span.hpp"
#include "util.hpp"

#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace mneme {

template <typename Storage, std::size_t Stride = dynamic_extent> class StridedView {
public:
    using offset_type = typename Storage::offset_type;
    using iterator = Iterator<StridedView<Storage, Stride>>;

    StridedView() : size_(0), container_(nullptr) {}

    template <class Layout>
    StridedView(Layout const& layout, std::shared_ptr<Storage> container, std::size_t from,
                std::size_t to) {
        setStorage(layout, std::move(container), from, to);
    }

    template <class Layout>
    void setStorage(Layout const& layout, std::shared_ptr<Storage> container, std::size_t from,
                    std::size_t to) {
        size_ = to - from;
        container_ = std::move(container);
        if (!container_) {
            return;
        }
        offset = container_->offset(layout[from]);
        if (!(to > from)) {
            throw std::runtime_error("'To' must be larger than 'from'.");
        }
        if constexpr (Stride == dynamic_extent) {
            stride = layout[from + 1] - layout[from];
        } else {
            stride = Stride;
        }
        for (std::size_t i = from; i < to; ++i) {
            auto stridei = layout[i + 1] - layout[i];
            if (stridei != stride) {
                std::stringstream ss;
                ss << "Failed to construct strided view: Stride " << stridei << " != " << stride
                   << " at " << from << ".";
                throw std::runtime_error(ss.str());
            }
        }
    }

    void setStorage(std::shared_ptr<Storage> container, std::size_t from, std::size_t to,
                    std::size_t strd = 1u) {
        size_ = to - from;
        container_ = std::move(container);
        if (!container_) {
            return;
        }
        if constexpr (Stride == dynamic_extent) {
            stride = strd;
        } else {
            stride = Stride;
        }
        assert(container_->size() % stride == 0);

        offset = container_->offset(from * stride);
    }

    auto operator[](std::size_t localId) noexcept -> typename Storage::template value_type<Stride> {
        return (*const_cast<const StridedView<Storage, Stride>*>(this))[localId];
    }

    auto operator[](std::size_t localId) const noexcept ->
        typename Storage::template value_type<Stride> {
        assert(container_ != nullptr);
        std::size_t s;
        if constexpr (Stride == dynamic_extent) {
            s = stride;
        } else {
            s = Stride;
        }
        std::size_t from = localId * s;
        return container_->template get<Stride>(offset, from, from + s);
    }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    Storage const& storage() const noexcept { return *container_; }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size()); }

private:
    std::size_t size_ = 0, stride;
    std::shared_ptr<Storage> container_;
    offset_type offset;
};

template <typename Storage> using DenseView = StridedView<Storage, 1u>;

template <typename Storage> class GeneralView {
public:
    using offset_type = typename Storage::offset_type;
    using iterator = Iterator<GeneralView<Storage>>;

    GeneralView() : size_(0), container_(nullptr) {}

    template <class Layout>
    GeneralView(Layout const& layout, std::shared_ptr<Storage> container, std::size_t from,
                std::size_t to) {
        setStorage(layout, std::move(container), from, to);
    }

    template <class Layout>
    void setStorage(Layout const& layout, std::shared_ptr<Storage> container, std::size_t from,
                    std::size_t to) {
        size_ = to - from;
        sl.clear();
        sl.reserve(size_);
        sl.insert(sl.begin(), layout.begin() + from, layout.begin() + to);
        container_ = std::move(container);
        if (container_ == nullptr) {
            return;
        }
        offset = container_->offset(layout[from]);
        if (!(to > from)) {
            throw std::runtime_error("'To' must be larger than 'from'.");
        }
    }

    auto operator[](std::size_t localId) noexcept ->
        typename Storage::template value_type<dynamic_extent> {
        return (*const_cast<const GeneralView<Storage>*>(this))[localId];
    }

    auto operator[](std::size_t localId) const noexcept ->
        typename Storage::template value_type<dynamic_extent> {
        assert(container_ != nullptr);
        return container_->template get<dynamic_extent>(offset, sl[localId], sl[localId + 1]);
    }

    std::size_t size() const noexcept { return size_; }
    Storage const& storage() const noexcept { return *container_; }

    iterator begin() { return iterator(this, 0); }
    iterator end() { return iterator(this, size()); }

private:
    std::size_t size_ = 0;
    std::vector<std::size_t> sl;
    std::shared_ptr<Storage> container_;
    offset_type offset;
};

template <typename MaybeStride = StaticNothing, typename MaybePlan = StaticNothing,
          typename MaybeStorage = StaticNothing, typename MaybeClusterId = StaticNothing>
class LayeredViewFactory {

public:
    LayeredViewFactory() = default;

    LayeredViewFactory(MaybePlan maybePlan, MaybeStorage maybeStorage,
                       MaybeClusterId maybeClusterId)
        : maybePlan(maybePlan), maybeStorage(maybeStorage), maybeClusterId(maybeClusterId) {}

    template <std::size_t Stride> [[nodiscard]] auto withStride() const {
        return LayeredViewFactory<std::integral_constant<std::size_t, Stride>, MaybePlan,
                                  MaybeStorage, MaybeClusterId>(maybePlan, maybeStorage,
                                                                maybeClusterId);
    }

    [[nodiscard]] auto withDynamicStride() const { return withStride<dynamic_extent>(); }

    template <typename LayeredPlanT> auto withPlan(LayeredPlanT& plan) const {
        const auto somePlan = StaticSome<LayeredPlanT>(plan);
        return LayeredViewFactory<MaybeStride, StaticSome<LayeredPlanT>, MaybeStorage,
                                  MaybeClusterId>(somePlan, maybeStorage, maybeClusterId);
    }

    template <typename StorageT> constexpr auto withStorage(StorageT storage) const {
        auto someStorage = StaticSome<StorageT>(storage);
        return LayeredViewFactory<MaybeStride, MaybePlan, StaticSome<StorageT>, MaybeClusterId>(
            maybePlan, std::move(someStorage), maybeClusterId);
    }
    [[nodiscard]] auto withClusterId(std::size_t clusterId) const {
        const auto someClusterId = StaticSome<std::size_t>(clusterId);
        return LayeredViewFactory<MaybeStride, MaybePlan, MaybeStorage, StaticSome<std::size_t>>(
            maybePlan, maybeStorage, someClusterId);
    }

    template <typename Layer, typename MaybePlan_ = MaybePlan>
    [[nodiscard]] std::pair<std::size_t, std::size_t> getFromToForLayer() const {
        if constexpr (std::is_base_of_v<CombinedLayeredPlanBase, typename MaybePlan_::type>) {
            static_assert(!std::is_same_v<MaybeClusterId, StaticNothing>,
                          "Cluster id has to be set when using CombinedLayer.");
            const auto& layer = maybePlan.value.template getLayer<Layer>(maybeClusterId.value);
            return {layer.offset, layer.offset + layer.numElements};
        } else {
            const auto& layer = maybePlan.value.template getLayer<Layer>();
            return {layer.offset, layer.offset + layer.numElements};
        }
    }

    template <
        typename Layer, typename MaybeStride_ = MaybeStride, typename MaybePlan_ = MaybePlan,
        typename MaybeStorage_ = MaybeStorage, typename MaybeClusterId_ = MaybeClusterId,
        typename std::enable_if<!std::is_same<MaybeStride_, StaticNothing>::value, int>::type = 0,
        typename std::enable_if<!std::is_same<MaybePlan_, StaticNothing>::value, int>::type = 0,
        typename std::enable_if<!std::is_same<MaybeStorage_, StaticNothing>::value, int>::type = 0,
        typename std::enable_if<std::is_same<MaybeClusterId_, StaticNothing>::value, int>::type = 0>
    [[nodiscard]] auto createStridedView() const {
        auto layout = maybePlan.value.getLayout();
        const auto [from, to] = getFromToForLayer<Layer>();

        return StridedView<typename MaybeStorage_::type::element_type, MaybeStride::value>(
            layout, (maybeStorage.value), from, to);
    }

    template <
        typename Layer, typename MaybeStride_ = MaybeStride, typename MaybePlan_ = MaybePlan,
        typename MaybeStorage_ = MaybeStorage,
        typename std::enable_if<std::is_same<MaybeStride_, StaticNothing>::value, int>::type = 0,
        typename std::enable_if<!std::is_same<MaybePlan_, StaticNothing>::value, int>::type = 0,
        typename std::enable_if<!std::is_same<MaybeStorage_, StaticNothing>::value, int>::type = 0>
    constexpr auto createDenseView() {
        auto layout = maybePlan.value.getLayout();
        const auto [from, to] = getFromToForLayer<Layer>();

        return DenseView<typename MaybeStorage_::type::element_type>(
            layout, std::move(maybeStorage.value), from, to);
    }

private:
    MaybePlan maybePlan;
    MaybeStorage maybeStorage;
    MaybeClusterId maybeClusterId;
};

constexpr auto createViewFactory() {
    return LayeredViewFactory<StaticNothing, StaticNothing, StaticNothing, StaticNothing>();
}

} // namespace mneme

#endif // MNEME_VIEW_H_
