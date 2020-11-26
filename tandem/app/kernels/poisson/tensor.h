#ifndef TNDM_POISSON_TENSOR_H_
#define TNDM_POISSON_TENSOR_H_
namespace tndm {
  namespace poisson {
    namespace tensor {
      struct A {
        constexpr static unsigned const Shape[2] = {6, 6};
        constexpr static unsigned const Size = 36;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct D_x {
        constexpr static unsigned const Shape[3] = {6, 2, 7};
        constexpr static unsigned const Size = 84;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct D_xi {
        constexpr static unsigned const Shape[3] = {6, 2, 7};
        constexpr static unsigned const Size = 84;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct E {
        constexpr static unsigned const Shape[2] = {6, 7};
        constexpr static unsigned const Size = 42;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct Em {
        constexpr static unsigned const Shape[2] = {7, 6};
        constexpr static unsigned const Size = 42;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct F_Q {
        constexpr static unsigned const Shape[1] = {7};
        constexpr static unsigned const Size = 7;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct G {
        constexpr static unsigned const Shape[3] = {2, 2, 7};
        constexpr static unsigned const Size = 28;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct J {
        constexpr static unsigned const Shape[1] = {7};
        constexpr static unsigned const Size = 7;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct K {
        constexpr static unsigned const Shape[1] = {6};
        constexpr static unsigned const Size = 6;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct K_Q {
        constexpr static unsigned const Shape[1] = {7};
        constexpr static unsigned const Size = 7;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct W {
        constexpr static unsigned const Shape[1] = {7};
        constexpr static unsigned const Size = 7;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct a {
        constexpr static unsigned const Shape[][2] = {{6, 6}, {6, 6}, {6, 6}, {6, 6}};
        constexpr static unsigned const Size[] = {36, 36, 36, 36};
        constexpr static unsigned index(unsigned i0, unsigned i1) {
          return 1*i0 + 2*i1;
        }
        constexpr static unsigned size(unsigned i0, unsigned i1) {
          return Size[index(i0, i1)];
        }
        template<typename T>
        struct Container {
          T data[4];
          Container() : data{} {}
          inline T& operator()(unsigned i0, unsigned i1) {
            return data[index(i0, i1)];
          }
          inline T const& operator()(unsigned i0, unsigned i1) const {
            return data[index(i0, i1)];
          }
        };
      };
      struct b {
        constexpr static unsigned const Shape[1] = {6};
        constexpr static unsigned const Size = 6;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct d_x {
        constexpr static unsigned const Shape[][3] = {{6, 2, 3}, {6, 2, 3}};
        constexpr static unsigned const Size[] = {36, 36};
        constexpr static unsigned index(unsigned i0) {
          return 1*i0;
        }
        constexpr static unsigned size(unsigned i0) {
          return Size[index(i0)];
        }
        template<typename T>
        struct Container {
          T data[2];
          Container() : data{} {}
          inline T& operator()(unsigned i0) {
            return data[index(i0)];
          }
          inline T const& operator()(unsigned i0) const {
            return data[index(i0)];
          }
        };
      };
      struct d_xi {
        constexpr static unsigned const Shape[][3] = {{6, 2, 3}, {6, 2, 3}};
        constexpr static unsigned const Size[] = {36, 36};
        constexpr static unsigned index(unsigned i0) {
          return 1*i0;
        }
        constexpr static unsigned size(unsigned i0) {
          return Size[index(i0)];
        }
        template<typename T>
        struct Container {
          T data[2];
          Container() : data{} {}
          inline T& operator()(unsigned i0) {
            return data[index(i0)];
          }
          inline T const& operator()(unsigned i0) const {
            return data[index(i0)];
          }
        };
      };
      struct e {
        constexpr static unsigned const Shape[][2] = {{6, 3}, {6, 3}};
        constexpr static unsigned const Size[] = {18, 18};
        constexpr static unsigned index(unsigned i0) {
          return 1*i0;
        }
        constexpr static unsigned size(unsigned i0) {
          return Size[index(i0)];
        }
        template<typename T>
        struct Container {
          T data[2];
          Container() : data{} {}
          inline T& operator()(unsigned i0) {
            return data[index(i0)];
          }
          inline T const& operator()(unsigned i0) const {
            return data[index(i0)];
          }
        };
      };
      struct e_q_T {
        constexpr static unsigned const Shape[2] = {3, 2};
        constexpr static unsigned const Size = 6;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct em {
        constexpr static unsigned const Shape[][2] = {{3, 6}, {3, 6}};
        constexpr static unsigned const Size[] = {18, 18};
        constexpr static unsigned index(unsigned i0) {
          return 1*i0;
        }
        constexpr static unsigned size(unsigned i0) {
          return Size[index(i0)];
        }
        template<typename T>
        struct Container {
          T data[2];
          Container() : data{} {}
          inline T& operator()(unsigned i0) {
            return data[index(i0)];
          }
          inline T const& operator()(unsigned i0) const {
            return data[index(i0)];
          }
        };
      };
      struct f_q {
        constexpr static unsigned const Shape[1] = {3};
        constexpr static unsigned const Size = 3;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct g {
        constexpr static unsigned const Shape[][3] = {{2, 2, 3}, {2, 2, 3}};
        constexpr static unsigned const Size[] = {12, 12};
        constexpr static unsigned index(unsigned i0) {
          return 1*i0;
        }
        constexpr static unsigned size(unsigned i0) {
          return Size[index(i0)];
        }
        template<typename T>
        struct Container {
          T data[2];
          Container() : data{} {}
          inline T& operator()(unsigned i0) {
            return data[index(i0)];
          }
          inline T const& operator()(unsigned i0) const {
            return data[index(i0)];
          }
        };
      };
      struct grad_u {
        constexpr static unsigned const Shape[2] = {2, 2};
        constexpr static unsigned const Size = 4;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct k {
        constexpr static unsigned const Shape[][1] = {{6}, {6}};
        constexpr static unsigned const Size[] = {6, 6};
        constexpr static unsigned index(unsigned i0) {
          return 1*i0;
        }
        constexpr static unsigned size(unsigned i0) {
          return Size[index(i0)];
        }
        template<typename T>
        struct Container {
          T data[2];
          Container() : data{} {}
          inline T& operator()(unsigned i0) {
            return data[index(i0)];
          }
          inline T const& operator()(unsigned i0) const {
            return data[index(i0)];
          }
        };
      };
      struct matMinv {
        constexpr static unsigned const Shape[2] = {6, 6};
        constexpr static unsigned const Size = 36;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct minv {
        constexpr static unsigned const Shape[2] = {2, 2};
        constexpr static unsigned const Size = 4;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct n {
        constexpr static unsigned const Shape[2] = {2, 3};
        constexpr static unsigned const Size = 6;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct nl {
        constexpr static unsigned const Shape[1] = {3};
        constexpr static unsigned const Size = 3;
        constexpr static unsigned size() {
          return Size;
        }
      };
      struct u {
        constexpr static unsigned const Shape[][1] = {{6}, {6}};
        constexpr static unsigned const Size[] = {6, 6};
        constexpr static unsigned index(unsigned i0) {
          return 1*i0;
        }
        constexpr static unsigned size(unsigned i0) {
          return Size[index(i0)];
        }
        template<typename T>
        struct Container {
          T data[2];
          Container() : data{} {}
          inline T& operator()(unsigned i0) {
            return data[index(i0)];
          }
          inline T const& operator()(unsigned i0) const {
            return data[index(i0)];
          }
        };
      };
      struct w {
        constexpr static unsigned const Shape[1] = {3};
        constexpr static unsigned const Size = 3;
        constexpr static unsigned size() {
          return Size;
        }
      };
    } // namespace tensor
  } // namespace poisson
} // namespace tndm
#endif
