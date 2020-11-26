#ifndef MNEME_TAGGED_TUPLE_H_
#define MNEME_TAGGED_TUPLE_H_

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace mneme {

namespace detail {
template <typename T, typename... Ts> struct index;
template <typename T, typename... Ts> struct index<T, T, Ts...> : std::integral_constant<int, 0> {};
template <typename T, typename U, typename... Ts>
struct index<T, U, Ts...> : std::integral_constant<int, index<T, Ts...>::value + 1> {};
template <typename T, typename... Ts> inline constexpr int index_v = index<T, Ts...>::value;

template <typename T> struct identity { using type = T; };

template <template <typename> typename type_transform, typename... Ids>
struct tt_impl : public std::tuple<typename type_transform<typename Ids::type>::type...> {
    using std::tuple<typename type_transform<typename Ids::type>::type...>::tuple;
    template <typename Id> auto&& get() noexcept {
        return std::get<detail::index_v<Id, Ids...>>(*this);
    }
    template <typename Id> auto&& get() const noexcept {
        return std::get<detail::index_v<Id, Ids...>>(*this);
    }

    template <typename Id>
    using element_t = typename std::tuple_element_t<
        detail::index_v<Id, Ids...>,
        std::tuple<typename type_transform<typename Ids::type>::type...>>;
};
} // namespace detail

template <typename... Ids> class tagged_tuple : public detail::tt_impl<detail::identity, Ids...> {};

} // namespace mneme

#endif // MNEME_TAGGED_TUPLE_H_
