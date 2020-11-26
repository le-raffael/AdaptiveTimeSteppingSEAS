#ifndef MNEME_ITERATOR_H_
#define MNEME_ITERATOR_H_

#include <cstddef>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

namespace mneme {

template <typename Container> class Iterator {
public:
    using iterator_category = std::bidirectional_iterator_tag;
    using access_type = decltype(std::declval<Container>().operator[](0));
    using value_type = std::remove_reference_t<access_type>;
    using difference_type = std::ptrdiff_t;
    using pointer = value_type*;
    using reference = value_type&;

    Iterator(Container* container, std::size_t position = 0)
        : container(container), pos(position) {}

    bool operator!=(Iterator const& other) {
        return pos != other.pos || container != other.container;
    }
    bool operator==(Iterator const& other) { return !(*this != other); }

    Iterator& operator++() {
        ++pos;
        return *this;
    }
    Iterator& operator--() {
        --pos;
        return *this;
    }
    Iterator operator++(int) {
        Iterator copy(container, pos);
        ++pos;
        return copy;
    }
    Iterator operator--(int) {
        Iterator copy(container, pos);
        --pos;
        return copy;
    }

    access_type operator*() { return (*container)[pos]; }

private:
    Container* const container = nullptr;
    std::size_t pos = std::numeric_limits<std::size_t>::max();
};

} // namespace mneme

#endif // MNEME_ITERATOR_H_
