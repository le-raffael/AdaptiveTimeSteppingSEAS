#ifndef TNDM_POISSON_KERNEL_H_
#define TNDM_POISSON_KERNEL_H_
#include <cmath>
#include <limits>
#include "yateto.h"
#include "tensor.h"
namespace tndm {
  namespace poisson {
    namespace kernel {
      struct project_K {
        constexpr static unsigned long const NonZeroFlops = 151;
        constexpr static unsigned long const HardwareFlops = 163;
        constexpr static unsigned long const TmpMemRequiredInBytes = 104;
        constexpr static unsigned long const TmpMaxMemRequiredInBytes = 104;

        double const* Em{};
        double* K{};
        double const* K_Q{};
        double const* W{};
        double const* matMinv{};


        void execute();
      };
    } // namespace kernel
    namespace kernel {
      struct assembleVolume {
        constexpr static unsigned long const NonZeroFlops = 1399;
        constexpr static unsigned long const HardwareFlops = 1526;
        constexpr static unsigned long const TmpMemRequiredInBytes = 784;
        constexpr static unsigned long const TmpMaxMemRequiredInBytes = 784;

        double* A{};
        double* D_x{};
        double const* D_xi{};
        double const* Em{};
        double const* G{};
        double const* J{};
        double const* K{};
        double const* W{};


        void execute();
      };
    } // namespace kernel
    namespace kernel {
      struct assembleFacetLocal {
        constexpr static unsigned long const NonZeroFlops = 1014;
        constexpr static unsigned long const HardwareFlops = 1017;
        constexpr static unsigned long const TmpMemRequiredInBytes = 192;
        constexpr static unsigned long const TmpMaxMemRequiredInBytes = 192;

        double c00 = std::numeric_limits<double>::signaling_NaN();
        double c10 = std::numeric_limits<double>::signaling_NaN();
        double c20 = std::numeric_limits<double>::signaling_NaN();
        double const* K{};
        tensor::a::Container<double*> a;
        tensor::d_x::Container<double*> d_x;
        tensor::d_xi::Container<double const*> d_xi;
        tensor::e::Container<double const*> e;
        tensor::em::Container<double const*> em;
        tensor::g::Container<double const*> g;
        double const* n{};
        double const* nl{};
        double const* w{};


        void execute();
      };
    } // namespace kernel
    namespace kernel {
      struct assembleFacetNeighbour {
        constexpr static unsigned long const NonZeroFlops = 2736;
        constexpr static unsigned long const HardwareFlops = 2667;
        constexpr static unsigned long const TmpMemRequiredInBytes = 192;
        constexpr static unsigned long const TmpMaxMemRequiredInBytes = 192;

        double c00 = std::numeric_limits<double>::signaling_NaN();
        double c01 = std::numeric_limits<double>::signaling_NaN();
        double c10 = std::numeric_limits<double>::signaling_NaN();
        double c11 = std::numeric_limits<double>::signaling_NaN();
        double c20 = std::numeric_limits<double>::signaling_NaN();
        double c21 = std::numeric_limits<double>::signaling_NaN();
        double const* K{};
        tensor::a::Container<double*> a;
        tensor::d_x::Container<double*> d_x;
        tensor::d_xi::Container<double const*> d_xi;
        tensor::e::Container<double const*> e;
        tensor::em::Container<double const*> em;
        tensor::g::Container<double const*> g;
        double const* n{};
        double const* nl{};
        double const* w{};


        void execute();
      };
    } // namespace kernel
    namespace kernel {
      struct rhsVolume {
        constexpr static unsigned long const NonZeroFlops = 98;
        constexpr static unsigned long const HardwareFlops = 98;
        constexpr static unsigned long const TmpMemRequiredInBytes = 112;
        constexpr static unsigned long const TmpMaxMemRequiredInBytes = 112;

        double const* E{};
        double const* F_Q{};
        double const* J{};
        double const* W{};
        double* b{};


        void execute();
      };
    } // namespace kernel
    namespace kernel {
      struct rhsFacet {
        constexpr static unsigned long const NonZeroFlops = 189;
        constexpr static unsigned long const HardwareFlops = 186;
        constexpr static unsigned long const TmpMemRequiredInBytes = 120;
        constexpr static unsigned long const TmpMaxMemRequiredInBytes = 120;

        double c10 = std::numeric_limits<double>::signaling_NaN();
        double c20 = std::numeric_limits<double>::signaling_NaN();
        double const* K{};
        double* b{};
        tensor::d_xi::Container<double const*> d_xi;
        tensor::e::Container<double const*> e;
        tensor::em::Container<double const*> em;
        double const* f_q{};
        tensor::g::Container<double const*> g;
        double const* n{};
        double const* nl{};
        double const* w{};


        void execute();
      };
    } // namespace kernel
    namespace kernel {
      struct grad_u {
        constexpr static unsigned long const NonZeroFlops = 486;
        constexpr static unsigned long const HardwareFlops = 574;
        constexpr static unsigned long const TmpMemRequiredInBytes = 144;
        constexpr static unsigned long const TmpMaxMemRequiredInBytes = 144;

        tensor::d_x::Container<double*> d_x;
        tensor::d_xi::Container<double const*> d_xi;
        double const* e_q_T{};
        tensor::em::Container<double const*> em;
        tensor::g::Container<double const*> g;
        double* grad_u{};
        tensor::k::Container<double const*> k;
        double const* minv{};
        tensor::u::Container<double const*> u;
        double const* w{};


        void execute();
      };
    } // namespace kernel
  } // namespace poisson
} // namespace tndm
#endif
