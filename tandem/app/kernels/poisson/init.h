#ifndef TNDM_POISSON_INIT_H_
#define TNDM_POISSON_INIT_H_
#include "tensor.h"
#include "yateto.h"
namespace tndm {
  namespace poisson {
    namespace init {
      struct A : tensor::A {
        constexpr static unsigned const Start[] = {0, 0};
        constexpr static unsigned const Stop[] = {6, 6};

        struct view {
          typedef ::yateto::DenseTensorView<2,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 6}, {0, 0}, {6, 6});
          }
        };
      };
      struct D_x : tensor::D_x {
        constexpr static unsigned const Start[] = {0, 0, 0};
        constexpr static unsigned const Stop[] = {6, 2, 7};

        struct view {
          typedef ::yateto::DenseTensorView<3,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<3,double,unsigned>(values, {6, 2, 7}, {0, 0, 0}, {6, 2, 7});
          }
        };
      };
      struct D_xi : tensor::D_xi {
        constexpr static unsigned const Start[] = {0, 0, 0};
        constexpr static unsigned const Stop[] = {6, 2, 7};

        struct view {
          typedef ::yateto::DenseTensorView<3,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<3,double,unsigned>(values, {6, 2, 7}, {0, 0, 0}, {6, 2, 7});
          }
        };
      };
      struct E : tensor::E {
        constexpr static unsigned const Start[] = {0, 0};
        constexpr static unsigned const Stop[] = {6, 7};

        struct view {
          typedef ::yateto::DenseTensorView<2,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 7}, {0, 0}, {6, 7});
          }
        };
      };
      struct Em : tensor::Em {
        constexpr static unsigned const Start[] = {0, 0};
        constexpr static unsigned const Stop[] = {7, 6};

        struct view {
          typedef ::yateto::DenseTensorView<2,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<2,double,unsigned>(values, {7, 6}, {0, 0}, {7, 6});
          }
        };
      };
      struct F_Q : tensor::F_Q {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {7};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {7}, {0}, {7});
          }
        };
      };
      struct G : tensor::G {
        constexpr static unsigned const Start[] = {0, 0, 0};
        constexpr static unsigned const Stop[] = {2, 2, 7};

        struct view {
          typedef ::yateto::DenseTensorView<3,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<3,double,unsigned>(values, {2, 2, 7}, {0, 0, 0}, {2, 2, 7});
          }
        };
      };
      struct J : tensor::J {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {7};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {7}, {0}, {7});
          }
        };
      };
      struct K : tensor::K {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {6};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {6}, {0}, {6});
          }
        };
      };
      struct K_Q : tensor::K_Q {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {7};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {7}, {0}, {7});
          }
        };
      };
      struct W : tensor::W {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {7};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {7}, {0}, {7});
          }
        };
      };
      struct a : tensor::a {
        constexpr static unsigned const Start0[] = {0, 0};
        constexpr static unsigned const Stop0[] = {6, 6};
        constexpr static unsigned const Start2[] = {0, 0};
        constexpr static unsigned const Stop2[] = {6, 6};
        constexpr static unsigned const Start1[] = {0, 0};
        constexpr static unsigned const Stop1[] = {6, 6};
        constexpr static unsigned const Start3[] = {0, 0};
        constexpr static unsigned const Stop3[] = {6, 6};

        template<unsigned i0, unsigned i1> struct view {};
      };
      template<>
      struct a::view<0,0> {
        typedef ::yateto::DenseTensorView<2,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 6}, {0, 0}, {6, 6});
        }
      };
      template<>
      struct a::view<0,1> {
        typedef ::yateto::DenseTensorView<2,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 6}, {0, 0}, {6, 6});
        }
      };
      template<>
      struct a::view<1,0> {
        typedef ::yateto::DenseTensorView<2,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 6}, {0, 0}, {6, 6});
        }
      };
      template<>
      struct a::view<1,1> {
        typedef ::yateto::DenseTensorView<2,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 6}, {0, 0}, {6, 6});
        }
      };
      struct b : tensor::b {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {6};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {6}, {0}, {6});
          }
        };
      };
      struct d_x : tensor::d_x {
        constexpr static unsigned const Start0[] = {0, 0, 0};
        constexpr static unsigned const Stop0[] = {6, 2, 3};
        constexpr static unsigned const Start1[] = {0, 0, 0};
        constexpr static unsigned const Stop1[] = {6, 2, 3};

        template<unsigned i0> struct view {};
      };
      template<>
      struct d_x::view<0> {
        typedef ::yateto::DenseTensorView<3,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<3,double,unsigned>(values, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
        }
      };
      template<>
      struct d_x::view<1> {
        typedef ::yateto::DenseTensorView<3,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<3,double,unsigned>(values, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
        }
      };
      struct d_xi : tensor::d_xi {
        constexpr static unsigned const Start0[] = {0, 0, 0};
        constexpr static unsigned const Stop0[] = {6, 2, 3};
        constexpr static unsigned const Start1[] = {0, 0, 0};
        constexpr static unsigned const Stop1[] = {6, 2, 3};

        template<unsigned i0> struct view {};
      };
      template<>
      struct d_xi::view<0> {
        typedef ::yateto::DenseTensorView<3,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<3,double,unsigned>(values, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
        }
      };
      template<>
      struct d_xi::view<1> {
        typedef ::yateto::DenseTensorView<3,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<3,double,unsigned>(values, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
        }
      };
      struct e : tensor::e {
        constexpr static unsigned const Start0[] = {0, 0};
        constexpr static unsigned const Stop0[] = {6, 3};
        constexpr static unsigned const Start1[] = {0, 0};
        constexpr static unsigned const Stop1[] = {6, 3};

        template<unsigned i0> struct view {};
      };
      template<>
      struct e::view<0> {
        typedef ::yateto::DenseTensorView<2,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 3}, {0, 0}, {6, 3});
        }
      };
      template<>
      struct e::view<1> {
        typedef ::yateto::DenseTensorView<2,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 3}, {0, 0}, {6, 3});
        }
      };
      struct e_q_T : tensor::e_q_T {
        constexpr static unsigned const Start[] = {0, 0};
        constexpr static unsigned const Stop[] = {3, 2};

        struct view {
          typedef ::yateto::DenseTensorView<2,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<2,double,unsigned>(values, {3, 2}, {0, 0}, {3, 2});
          }
        };
      };
      struct em : tensor::em {
        constexpr static unsigned const Start0[] = {0, 0};
        constexpr static unsigned const Stop0[] = {3, 6};
        constexpr static unsigned const Start1[] = {0, 0};
        constexpr static unsigned const Stop1[] = {3, 6};

        template<unsigned i0> struct view {};
      };
      template<>
      struct em::view<0> {
        typedef ::yateto::DenseTensorView<2,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<2,double,unsigned>(values, {3, 6}, {0, 0}, {3, 6});
        }
      };
      template<>
      struct em::view<1> {
        typedef ::yateto::DenseTensorView<2,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<2,double,unsigned>(values, {3, 6}, {0, 0}, {3, 6});
        }
      };
      struct f_q : tensor::f_q {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {3};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {3}, {0}, {3});
          }
        };
      };
      struct g : tensor::g {
        constexpr static unsigned const Start0[] = {0, 0, 0};
        constexpr static unsigned const Stop0[] = {2, 2, 3};
        constexpr static unsigned const Start1[] = {0, 0, 0};
        constexpr static unsigned const Stop1[] = {2, 2, 3};

        template<unsigned i0> struct view {};
      };
      template<>
      struct g::view<0> {
        typedef ::yateto::DenseTensorView<3,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<3,double,unsigned>(values, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
        }
      };
      template<>
      struct g::view<1> {
        typedef ::yateto::DenseTensorView<3,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<3,double,unsigned>(values, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
        }
      };
      struct grad_u : tensor::grad_u {
        constexpr static unsigned const Start[] = {0, 0};
        constexpr static unsigned const Stop[] = {2, 2};

        struct view {
          typedef ::yateto::DenseTensorView<2,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<2,double,unsigned>(values, {2, 2}, {0, 0}, {2, 2});
          }
        };
      };
      struct k : tensor::k {
        constexpr static unsigned const Start0[] = {0};
        constexpr static unsigned const Stop0[] = {6};
        constexpr static unsigned const Start1[] = {0};
        constexpr static unsigned const Stop1[] = {6};

        template<unsigned i0> struct view {};
      };
      template<>
      struct k::view<0> {
        typedef ::yateto::DenseTensorView<1,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<1,double,unsigned>(values, {6}, {0}, {6});
        }
      };
      template<>
      struct k::view<1> {
        typedef ::yateto::DenseTensorView<1,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<1,double,unsigned>(values, {6}, {0}, {6});
        }
      };
      struct matMinv : tensor::matMinv {
        constexpr static unsigned const Start[] = {0, 0};
        constexpr static unsigned const Stop[] = {6, 6};

        struct view {
          typedef ::yateto::DenseTensorView<2,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<2,double,unsigned>(values, {6, 6}, {0, 0}, {6, 6});
          }
        };
      };
      struct minv : tensor::minv {
        constexpr static unsigned const Start[] = {0, 0};
        constexpr static unsigned const Stop[] = {2, 2};

        struct view {
          typedef ::yateto::DenseTensorView<2,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<2,double,unsigned>(values, {2, 2}, {0, 0}, {2, 2});
          }
        };
      };
      struct n : tensor::n {
        constexpr static unsigned const Start[] = {0, 0};
        constexpr static unsigned const Stop[] = {2, 3};

        struct view {
          typedef ::yateto::DenseTensorView<2,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<2,double,unsigned>(values, {2, 3}, {0, 0}, {2, 3});
          }
        };
      };
      struct nl : tensor::nl {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {3};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {3}, {0}, {3});
          }
        };
      };
      struct u : tensor::u {
        constexpr static unsigned const Start0[] = {0};
        constexpr static unsigned const Stop0[] = {6};
        constexpr static unsigned const Start1[] = {0};
        constexpr static unsigned const Stop1[] = {6};

        template<unsigned i0> struct view {};
      };
      template<>
      struct u::view<0> {
        typedef ::yateto::DenseTensorView<1,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<1,double,unsigned>(values, {6}, {0}, {6});
        }
      };
      template<>
      struct u::view<1> {
        typedef ::yateto::DenseTensorView<1,double,unsigned> type;
        static inline type create(double* values) {
          return ::yateto::DenseTensorView<1,double,unsigned>(values, {6}, {0}, {6});
        }
      };
      struct w : tensor::w {
        constexpr static unsigned const Start[] = {0};
        constexpr static unsigned const Stop[] = {3};

        struct view {
          typedef ::yateto::DenseTensorView<1,double,unsigned> type;
          static inline type create(double* values) {
            return ::yateto::DenseTensorView<1,double,unsigned>(values, {3}, {0}, {3});
          }
        };
      };
    } // namespace init
  } // namespace poisson
} // namespace tndm
#endif
