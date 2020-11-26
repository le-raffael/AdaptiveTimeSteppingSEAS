#ifndef TNDM_POISSON_KERNELTEST_T_H_
#define TNDM_POISSON_KERNELTEST_T_H_
#include "kernel.h"
#include "init.h"
#include "yateto.h"
#ifndef NDEBUG
long long libxsmm_num_total_flops = 0;
long long pspamm_num_total_flops = 0;
#endif
#include <cxxtest/TestSuite.h>
namespace tndm {
  namespace poisson {
    namespace unit_test {
      class KernelTestSuite;
    } // namespace unit_test
  } // namespace poisson
} // namespace tndm
class tndm::poisson::unit_test::KernelTestSuite : public CxxTest::TestSuite {
public:
  void testproject_K() {
    alignas(16) double Em[42] ;
    for (int i = 0; i < 42; ++i) {
      Em[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_Em[42]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_Em(_ut_Em, {7, 6}, {0, 0}, {7, 6});
    init::Em::view::create(Em).copyToView(_view__ut_Em);

    alignas(16) double K[6] ;
    for (int i = 0; i < 6; ++i) {
      K[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_K[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_K(_ut_K, {6}, {0}, {6});
    init::K::view::create(K).copyToView(_view__ut_K);

    alignas(16) double K_Q[7] ;
    for (int i = 0; i < 7; ++i) {
      K_Q[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_K_Q[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_K_Q(_ut_K_Q, {7}, {0}, {7});
    init::K_Q::view::create(K_Q).copyToView(_view__ut_K_Q);

    alignas(16) double W[7] ;
    for (int i = 0; i < 7; ++i) {
      W[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_W[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_W(_ut_W, {7}, {0}, {7});
    init::W::view::create(W).copyToView(_view__ut_W);

    alignas(16) double matMinv[36] ;
    for (int i = 0; i < 36; ++i) {
      matMinv[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_matMinv[36]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_matMinv(_ut_matMinv, {6, 6}, {0, 0}, {6, 6});
    init::matMinv::view::create(matMinv).copyToView(_view__ut_matMinv);

    kernel::project_K krnl;
    krnl.Em = Em;
    krnl.K = K;
    krnl.K_Q = K_Q;
    krnl.W = W;
    krnl.matMinv = matMinv;
    krnl.execute();

    double *_tmp0;
    alignas(16) double _buffer0[6] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 6 * sizeof(double));
    for (int _q = 0; _q < 7; ++_q) {
      for (int _k = 0; _k < 6; ++_k) {
        for (int _p = 0; _p < 6; ++_p) {
          _tmp0[1*_p] += _ut_matMinv[1*_p + 6*_k] * _ut_K_Q[1*_q] * _ut_Em[1*_q + 7*_k] * _ut_W[1*_q];
        }
      }
    }
    for (int _a = 0; _a < 6; ++_a) {
      _ut_K[1*_a] = _tmp0[1*_a];
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 6; ++_a) {
        double ref = _ut_K[1*_a];
        double diff = ref - K[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testassembleVolume() {
    alignas(16) double A[36] ;
    for (int i = 0; i < 36; ++i) {
      A[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_A[36]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_A(_ut_A, {6, 6}, {0, 0}, {6, 6});
    init::A::view::create(A).copyToView(_view__ut_A);

    alignas(16) double D_x[84] ;
    for (int i = 0; i < 84; ++i) {
      D_x[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_D_x[84]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_D_x(_ut_D_x, {6, 2, 7}, {0, 0, 0}, {6, 2, 7});
    init::D_x::view::create(D_x).copyToView(_view__ut_D_x);

    alignas(16) double D_xi[84] ;
    for (int i = 0; i < 84; ++i) {
      D_xi[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_D_xi[84]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_D_xi(_ut_D_xi, {6, 2, 7}, {0, 0, 0}, {6, 2, 7});
    init::D_xi::view::create(D_xi).copyToView(_view__ut_D_xi);

    alignas(16) double Em[42] ;
    for (int i = 0; i < 42; ++i) {
      Em[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_Em[42]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_Em(_ut_Em, {7, 6}, {0, 0}, {7, 6});
    init::Em::view::create(Em).copyToView(_view__ut_Em);

    alignas(16) double G[28] ;
    for (int i = 0; i < 28; ++i) {
      G[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_G[28]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_G(_ut_G, {2, 2, 7}, {0, 0, 0}, {2, 2, 7});
    init::G::view::create(G).copyToView(_view__ut_G);

    alignas(16) double J[7] ;
    for (int i = 0; i < 7; ++i) {
      J[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_J(_ut_J, {7}, {0}, {7});
    init::J::view::create(J).copyToView(_view__ut_J);

    alignas(16) double K[6] ;
    for (int i = 0; i < 6; ++i) {
      K[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_K[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_K(_ut_K, {6}, {0}, {6});
    init::K::view::create(K).copyToView(_view__ut_K);

    alignas(16) double W[7] ;
    for (int i = 0; i < 7; ++i) {
      W[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_W[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_W(_ut_W, {7}, {0}, {7});
    init::W::view::create(W).copyToView(_view__ut_W);

    kernel::assembleVolume krnl;
    krnl.A = A;
    krnl.D_x = D_x;
    krnl.D_xi = D_xi;
    krnl.Em = Em;
    krnl.G = G;
    krnl.J = J;
    krnl.K = K;
    krnl.W = W;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2;
    alignas(16) double _buffer0[84] ;
    alignas(16) double _buffer1[84] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 84 * sizeof(double));
    for (int _e = 0; _e < 2; ++_e) {
      for (int _q = 0; _q < 7; ++_q) {
        for (int _k = 0; _k < 6; ++_k) {
          for (int _i = 0; _i < 2; ++_i) {
            _tmp0[1*_i + 2*_k + 12*_q] += _ut_G[1*_e + 2*_i + 4*_q] * _ut_D_xi[1*_k + 6*_e + 12*_q];
          }
        }
      }
    }
    _tmp1 = _buffer1;
    for (int _q = 0; _q < 7; ++_q) {
      for (int _i = 0; _i < 2; ++_i) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp1[1*_k + 6*_i + 12*_q] = _tmp0[1*_i + 2*_k + 12*_q];
        }
      }
    }
    for (int _c = 0; _c < 7; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _ut_D_x[1*_a + 6*_b + 12*_c] = _tmp1[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp2 = _buffer0;
    memset(_tmp2, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _m = 0; _m < 6; ++_m) {
        for (int _q = 0; _q < 7; ++_q) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp2[1*_k + 6*_l] += _ut_J[1*_q] * _ut_W[1*_q] * _ut_K[1*_m] * _ut_Em[1*_q + 7*_m] * _ut_D_x[1*_k + 6*_i + 12*_q] * _ut_D_x[1*_l + 6*_i + 12*_q];
            }
          }
        }
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_A[1*_a + 6*_b] = _tmp2[1*_a + 6*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 6; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_A[1*_a + 6*_b];
          double diff = ref - A[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _c = 0; _c < 7; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            double ref = _ut_D_x[1*_a + 6*_b + 12*_c];
            double diff = ref - D_x[1*_a + 6*_b + 12*_c];
            error += diff * diff;
            refNorm += ref * ref;
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testassembleFacetLocal() {
    double c00 = 2.0;
    double c10 = 3.0;
    double c20 = 4.0;
    alignas(16) double K[6] ;
    for (int i = 0; i < 6; ++i) {
      K[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_K[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_K(_ut_K, {6}, {0}, {6});
    init::K::view::create(K).copyToView(_view__ut_K);

    alignas(16) double a_0_0[36] ;
    for (int i = 0; i < 36; ++i) {
      a_0_0[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_a_0_0[36]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_a_0_0(_ut_a_0_0, {6, 6}, {0, 0}, {6, 6});
    init::a::view<0,0>::create(a_0_0).copyToView(_view__ut_a_0_0);

    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

    alignas(16) double d_xi_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_xi_0[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_d_xi_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_xi_0(_ut_d_xi_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_xi::view<0>::create(d_xi_0).copyToView(_view__ut_d_xi_0);

    alignas(16) double e_0[18] ;
    for (int i = 0; i < 18; ++i) {
      e_0[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_e_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_0(_ut_e_0, {6, 3}, {0, 0}, {6, 3});
    init::e::view<0>::create(e_0).copyToView(_view__ut_e_0);

    alignas(16) double em_0[18] ;
    for (int i = 0; i < 18; ++i) {
      em_0[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_em_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_em_0(_ut_em_0, {3, 6}, {0, 0}, {3, 6});
    init::em::view<0>::create(em_0).copyToView(_view__ut_em_0);

    alignas(16) double g_0[12] ;
    for (int i = 0; i < 12; ++i) {
      g_0[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_g_0[12]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_g_0(_ut_g_0, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
    init::g::view<0>::create(g_0).copyToView(_view__ut_g_0);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::assembleFacetLocal krnl;
    krnl.c00 = c00;
    krnl.c10 = c10;
    krnl.c20 = c20;
    krnl.K = K;
    krnl.a(0,0) = a_0_0;
    krnl.d_x(0) = d_x_0;
    krnl.d_xi(0) = d_xi_0;
    krnl.e(0) = e_0;
    krnl.em(0) = em_0;
    krnl.g(0) = g_0;
    krnl.n = n;
    krnl.nl = nl;
    krnl.w = w;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8;
    alignas(16) double _buffer0[36] ;
    alignas(16) double _buffer1[36] ;
    alignas(16) double _buffer2[36] ;
    alignas(16) double _buffer3[36] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 36 * sizeof(double));
    for (int _e = 0; _e < 2; ++_e) {
      for (int _m = 0; _m < 6; ++_m) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _i = 0; _i < 2; ++_i) {
              _tmp0[1*_i + 2*_k + 12*_q] += _ut_K[1*_m] * _ut_em_0[1*_q + 3*_m] * _ut_g_0[1*_e + 2*_i + 4*_q] * _ut_d_xi_0[1*_k + 6*_e + 12*_q];
            }
          }
        }
      }
    }
    _tmp1 = _buffer1;
    for (int _q = 0; _q < 3; ++_q) {
      for (int _i = 0; _i < 2; ++_i) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp1[1*_k + 6*_i + 12*_q] = _tmp0[1*_i + 2*_k + 12*_q];
        }
      }
    }
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _ut_d_x_0[1*_a + 6*_b + 12*_c] = _tmp1[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp2 = _buffer0;
    memset(_tmp2, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _l = 0; _l < 6; ++_l) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp2[1*_k + 6*_l] += _ut_w[1*_q] * _ut_d_x_0[1*_k + 6*_i + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_e_0[1*_l + 6*_q];
          }
        }
      }
    }
    _tmp3 = _buffer1;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp3[1*_a + 6*_b] = c00 * _tmp2[1*_a + 6*_b];
      }
    }
    _tmp4 = _buffer0;
    memset(_tmp4, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _l = 0; _l < 6; ++_l) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp4[1*_k + 6*_l] += _ut_w[1*_q] * _ut_d_x_0[1*_l + 6*_i + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_e_0[1*_k + 6*_q];
          }
        }
      }
    }
    _tmp5 = _buffer2;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp5[1*_a + 6*_b] = c10 * _tmp4[1*_a + 6*_b];
      }
    }
    _tmp6 = _buffer0;
    memset(_tmp6, 0, 36 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp6[1*_k + 6*_l] += _ut_w[1*_q] * _ut_e_0[1*_k + 6*_q] * _ut_e_0[1*_l + 6*_q] * _ut_nl[1*_q];
        }
      }
    }
    _tmp7 = _buffer3;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp7[1*_a + 6*_b] = c20 * _tmp6[1*_a + 6*_b];
      }
    }
    _tmp8 = _buffer0;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp8[1*_a + 6*_b] = _tmp3[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp8[1*_a + 6*_b] += _tmp5[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp8[1*_a + 6*_b] += _tmp7[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_a_0_0[1*_a + 6*_b] = _tmp8[1*_a + 6*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 6; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_a_0_0[1*_a + 6*_b];
          double diff = ref - a_0_0[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            double ref = _ut_d_x_0[1*_a + 6*_b + 12*_c];
            double diff = ref - d_x_0[1*_a + 6*_b + 12*_c];
            error += diff * diff;
            refNorm += ref * ref;
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testassembleFacetNeighbour() {
    double c00 = 2.0;
    double c01 = 3.0;
    double c10 = 4.0;
    double c11 = 5.0;
    double c20 = 6.0;
    double c21 = 7.0;
    alignas(16) double K[6] ;
    for (int i = 0; i < 6; ++i) {
      K[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_K[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_K(_ut_K, {6}, {0}, {6});
    init::K::view::create(K).copyToView(_view__ut_K);

    alignas(16) double a_0_1[36] ;
    for (int i = 0; i < 36; ++i) {
      a_0_1[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_a_0_1[36]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_a_0_1(_ut_a_0_1, {6, 6}, {0, 0}, {6, 6});
    init::a::view<0,1>::create(a_0_1).copyToView(_view__ut_a_0_1);

    alignas(16) double a_1_0[36] ;
    for (int i = 0; i < 36; ++i) {
      a_1_0[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_a_1_0[36]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_a_1_0(_ut_a_1_0, {6, 6}, {0, 0}, {6, 6});
    init::a::view<1,0>::create(a_1_0).copyToView(_view__ut_a_1_0);

    alignas(16) double a_1_1[36] ;
    for (int i = 0; i < 36; ++i) {
      a_1_1[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_a_1_1[36]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_a_1_1(_ut_a_1_1, {6, 6}, {0, 0}, {6, 6});
    init::a::view<1,1>::create(a_1_1).copyToView(_view__ut_a_1_1);

    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

    alignas(16) double d_x_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_1[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_d_x_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_1(_ut_d_x_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<1>::create(d_x_1).copyToView(_view__ut_d_x_1);

    alignas(16) double d_xi_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_xi_1[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_d_xi_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_xi_1(_ut_d_xi_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_xi::view<1>::create(d_xi_1).copyToView(_view__ut_d_xi_1);

    alignas(16) double e_0[18] ;
    for (int i = 0; i < 18; ++i) {
      e_0[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_e_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_0(_ut_e_0, {6, 3}, {0, 0}, {6, 3});
    init::e::view<0>::create(e_0).copyToView(_view__ut_e_0);

    alignas(16) double e_1[18] ;
    for (int i = 0; i < 18; ++i) {
      e_1[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_e_1[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_1(_ut_e_1, {6, 3}, {0, 0}, {6, 3});
    init::e::view<1>::create(e_1).copyToView(_view__ut_e_1);

    alignas(16) double em_1[18] ;
    for (int i = 0; i < 18; ++i) {
      em_1[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_em_1[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_em_1(_ut_em_1, {3, 6}, {0, 0}, {3, 6});
    init::em::view<1>::create(em_1).copyToView(_view__ut_em_1);

    alignas(16) double g_1[12] ;
    for (int i = 0; i < 12; ++i) {
      g_1[i] = static_cast<double>((i + 10) % 512 + 1);
    }
    alignas(16) double _ut_g_1[12]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_g_1(_ut_g_1, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
    init::g::view<1>::create(g_1).copyToView(_view__ut_g_1);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 11) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 12) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 13) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::assembleFacetNeighbour krnl;
    krnl.c00 = c00;
    krnl.c01 = c01;
    krnl.c10 = c10;
    krnl.c11 = c11;
    krnl.c20 = c20;
    krnl.c21 = c21;
    krnl.K = K;
    krnl.a(0,1) = a_0_1;
    krnl.a(1,0) = a_1_0;
    krnl.a(1,1) = a_1_1;
    krnl.d_x(0) = d_x_0;
    krnl.d_x(1) = d_x_1;
    krnl.d_xi(1) = d_xi_1;
    krnl.e(0) = e_0;
    krnl.e(1) = e_1;
    krnl.em(1) = em_1;
    krnl.g(1) = g_1;
    krnl.n = n;
    krnl.nl = nl;
    krnl.w = w;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11, *_tmp12, *_tmp13, *_tmp14, *_tmp15, *_tmp16, *_tmp17, *_tmp18, *_tmp19, *_tmp20, *_tmp21, *_tmp22;
    alignas(16) double _buffer0[36] ;
    alignas(16) double _buffer1[36] ;
    alignas(16) double _buffer2[36] ;
    alignas(16) double _buffer3[36] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 36 * sizeof(double));
    for (int _e = 0; _e < 2; ++_e) {
      for (int _m = 0; _m < 6; ++_m) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _i = 0; _i < 2; ++_i) {
              _tmp0[1*_i + 2*_k + 12*_q] += _ut_K[1*_m] * _ut_em_1[1*_q + 3*_m] * _ut_g_1[1*_e + 2*_i + 4*_q] * _ut_d_xi_1[1*_k + 6*_e + 12*_q];
            }
          }
        }
      }
    }
    _tmp1 = _buffer1;
    for (int _q = 0; _q < 3; ++_q) {
      for (int _i = 0; _i < 2; ++_i) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp1[1*_k + 6*_i + 12*_q] = _tmp0[1*_i + 2*_k + 12*_q];
        }
      }
    }
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _ut_d_x_1[1*_a + 6*_b + 12*_c] = _tmp1[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp2 = _buffer0;
    memset(_tmp2, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _l = 0; _l < 6; ++_l) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp2[1*_k + 6*_l] += _ut_w[1*_q] * _ut_d_x_0[1*_k + 6*_i + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_e_1[1*_l + 6*_q];
          }
        }
      }
    }
    _tmp3 = _buffer1;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp3[1*_a + 6*_b] = c01 * _tmp2[1*_a + 6*_b];
      }
    }
    _tmp4 = _buffer0;
    memset(_tmp4, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _l = 0; _l < 6; ++_l) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp4[1*_k + 6*_l] += _ut_w[1*_q] * _ut_d_x_1[1*_l + 6*_i + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_e_0[1*_k + 6*_q];
          }
        }
      }
    }
    _tmp5 = _buffer2;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp5[1*_a + 6*_b] = c10 * _tmp4[1*_a + 6*_b];
      }
    }
    _tmp6 = _buffer0;
    memset(_tmp6, 0, 36 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp6[1*_k + 6*_l] += _ut_w[1*_q] * _ut_e_0[1*_k + 6*_q] * _ut_e_1[1*_l + 6*_q] * _ut_nl[1*_q];
        }
      }
    }
    _tmp7 = _buffer3;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp7[1*_a + 6*_b] = c21 * _tmp6[1*_a + 6*_b];
      }
    }
    _tmp8 = _buffer0;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp8[1*_a + 6*_b] = _tmp3[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp8[1*_a + 6*_b] += _tmp5[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp8[1*_a + 6*_b] += _tmp7[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_a_0_1[1*_a + 6*_b] = _tmp8[1*_a + 6*_b];
      }
    }
    _tmp9 = _buffer1;
    memset(_tmp9, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _l = 0; _l < 6; ++_l) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp9[1*_k + 6*_l] += _ut_w[1*_q] * _ut_d_x_1[1*_k + 6*_i + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_e_0[1*_l + 6*_q];
          }
        }
      }
    }
    _tmp10 = _buffer2;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp10[1*_a + 6*_b] = c00 * _tmp9[1*_a + 6*_b];
      }
    }
    _tmp11 = _buffer3;
    memset(_tmp11, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _l = 0; _l < 6; ++_l) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp11[1*_k + 6*_l] += _ut_w[1*_q] * _ut_d_x_0[1*_l + 6*_i + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_e_1[1*_k + 6*_q];
          }
        }
      }
    }
    _tmp12 = _buffer0;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp12[1*_a + 6*_b] = c11 * _tmp11[1*_a + 6*_b];
      }
    }
    _tmp13 = _buffer1;
    memset(_tmp13, 0, 36 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp13[1*_k + 6*_l] += _ut_w[1*_q] * _ut_e_1[1*_k + 6*_q] * _ut_e_0[1*_l + 6*_q] * _ut_nl[1*_q];
        }
      }
    }
    _tmp14 = _buffer3;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp14[1*_a + 6*_b] = c21 * _tmp13[1*_a + 6*_b];
      }
    }
    _tmp15 = _buffer1;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp15[1*_a + 6*_b] = _tmp10[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp15[1*_a + 6*_b] += _tmp12[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp15[1*_a + 6*_b] += _tmp14[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_a_1_0[1*_a + 6*_b] = _tmp15[1*_a + 6*_b];
      }
    }
    _tmp16 = _buffer2;
    memset(_tmp16, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _l = 0; _l < 6; ++_l) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp16[1*_k + 6*_l] += _ut_w[1*_q] * _ut_d_x_1[1*_k + 6*_i + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_e_1[1*_l + 6*_q];
          }
        }
      }
    }
    _tmp17 = _buffer0;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp17[1*_a + 6*_b] = c01 * _tmp16[1*_a + 6*_b];
      }
    }
    _tmp18 = _buffer3;
    memset(_tmp18, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _l = 0; _l < 6; ++_l) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp18[1*_k + 6*_l] += _ut_w[1*_q] * _ut_d_x_1[1*_l + 6*_i + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_e_1[1*_k + 6*_q];
          }
        }
      }
    }
    _tmp19 = _buffer1;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp19[1*_a + 6*_b] = c11 * _tmp18[1*_a + 6*_b];
      }
    }
    _tmp20 = _buffer2;
    memset(_tmp20, 0, 36 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp20[1*_k + 6*_l] += _ut_w[1*_q] * _ut_e_1[1*_k + 6*_q] * _ut_e_1[1*_l + 6*_q] * _ut_nl[1*_q];
        }
      }
    }
    _tmp21 = _buffer3;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp21[1*_a + 6*_b] = c20 * _tmp20[1*_a + 6*_b];
      }
    }
    _tmp22 = _buffer2;
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp22[1*_a + 6*_b] = _tmp17[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp22[1*_a + 6*_b] += _tmp19[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp22[1*_a + 6*_b] += _tmp21[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 6; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_a_1_1[1*_a + 6*_b] = _tmp22[1*_a + 6*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 6; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_a_0_1[1*_a + 6*_b];
          double diff = ref - a_0_1[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 6; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_a_1_0[1*_a + 6*_b];
          double diff = ref - a_1_0[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 6; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_a_1_1[1*_a + 6*_b];
          double diff = ref - a_1_1[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            double ref = _ut_d_x_1[1*_a + 6*_b + 12*_c];
            double diff = ref - d_x_1[1*_a + 6*_b + 12*_c];
            error += diff * diff;
            refNorm += ref * ref;
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testrhsVolume() {
    alignas(16) double E[42] ;
    for (int i = 0; i < 42; ++i) {
      E[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_E[42]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_E(_ut_E, {6, 7}, {0, 0}, {6, 7});
    init::E::view::create(E).copyToView(_view__ut_E);

    alignas(16) double F_Q[7] ;
    for (int i = 0; i < 7; ++i) {
      F_Q[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_F_Q[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_F_Q(_ut_F_Q, {7}, {0}, {7});
    init::F_Q::view::create(F_Q).copyToView(_view__ut_F_Q);

    alignas(16) double J[7] ;
    for (int i = 0; i < 7; ++i) {
      J[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_J(_ut_J, {7}, {0}, {7});
    init::J::view::create(J).copyToView(_view__ut_J);

    alignas(16) double W[7] ;
    for (int i = 0; i < 7; ++i) {
      W[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_W[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_W(_ut_W, {7}, {0}, {7});
    init::W::view::create(W).copyToView(_view__ut_W);

    alignas(16) double b[6] ;
    for (int i = 0; i < 6; ++i) {
      b[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_b[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_b(_ut_b, {6}, {0}, {6});
    init::b::view::create(b).copyToView(_view__ut_b);

    kernel::rhsVolume krnl;
    krnl.E = E;
    krnl.F_Q = F_Q;
    krnl.J = J;
    krnl.W = W;
    krnl.b = b;
    krnl.execute();

    double *_tmp0, *_tmp1;
    alignas(16) double _buffer0[6] ;
    alignas(16) double _buffer1[6] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 6 * sizeof(double));
    for (int _q = 0; _q < 7; ++_q) {
      for (int _k = 0; _k < 6; ++_k) {
        _tmp0[1*_k] += _ut_J[1*_q] * _ut_W[1*_q] * _ut_E[1*_k + 6*_q] * _ut_F_Q[1*_q];
      }
    }
    _tmp1 = _buffer1;
    for (int _a = 0; _a < 6; ++_a) {
      _tmp1[1*_a] = _ut_b[1*_a];
    }
    for (int _a = 0; _a < 6; ++_a) {
      _tmp1[1*_a] += _tmp0[1*_a];
    }
    for (int _a = 0; _a < 6; ++_a) {
      _ut_b[1*_a] = _tmp1[1*_a];
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 6; ++_a) {
        double ref = _ut_b[1*_a];
        double diff = ref - b[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testrhsFacet() {
    double c10 = 2.0;
    double c20 = 3.0;
    alignas(16) double K[6] ;
    for (int i = 0; i < 6; ++i) {
      K[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_K[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_K(_ut_K, {6}, {0}, {6});
    init::K::view::create(K).copyToView(_view__ut_K);

    alignas(16) double b[6] ;
    for (int i = 0; i < 6; ++i) {
      b[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_b[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_b(_ut_b, {6}, {0}, {6});
    init::b::view::create(b).copyToView(_view__ut_b);

    alignas(16) double d_xi_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_xi_0[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_d_xi_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_xi_0(_ut_d_xi_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_xi::view<0>::create(d_xi_0).copyToView(_view__ut_d_xi_0);

    alignas(16) double e_0[18] ;
    for (int i = 0; i < 18; ++i) {
      e_0[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_e_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_0(_ut_e_0, {6, 3}, {0, 0}, {6, 3});
    init::e::view<0>::create(e_0).copyToView(_view__ut_e_0);

    alignas(16) double em_0[18] ;
    for (int i = 0; i < 18; ++i) {
      em_0[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_em_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_em_0(_ut_em_0, {3, 6}, {0, 0}, {3, 6});
    init::em::view<0>::create(em_0).copyToView(_view__ut_em_0);

    alignas(16) double f_q[3] ;
    for (int i = 0; i < 3; ++i) {
      f_q[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_f_q[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_f_q(_ut_f_q, {3}, {0}, {3});
    init::f_q::view::create(f_q).copyToView(_view__ut_f_q);

    alignas(16) double g_0[12] ;
    for (int i = 0; i < 12; ++i) {
      g_0[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_g_0[12]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_g_0(_ut_g_0, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
    init::g::view<0>::create(g_0).copyToView(_view__ut_g_0);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::rhsFacet krnl;
    krnl.c10 = c10;
    krnl.c20 = c20;
    krnl.K = K;
    krnl.b = b;
    krnl.d_xi(0) = d_xi_0;
    krnl.e(0) = e_0;
    krnl.em(0) = em_0;
    krnl.f_q = f_q;
    krnl.g(0) = g_0;
    krnl.n = n;
    krnl.nl = nl;
    krnl.w = w;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4;
    alignas(16) double _buffer0[6] ;
    alignas(16) double _buffer1[6] ;
    alignas(16) double _buffer2[6] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 6 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _e = 0; _e < 2; ++_e) {
        for (int _m = 0; _m < 6; ++_m) {
          for (int _q = 0; _q < 3; ++_q) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp0[1*_k] += _ut_w[1*_q] * _ut_K[1*_m] * _ut_em_0[1*_q + 3*_m] * _ut_g_0[1*_e + 2*_i + 4*_q] * _ut_d_xi_0[1*_k + 6*_e + 12*_q] * _ut_n[1*_i + 2*_q] * _ut_f_q[1*_q];
            }
          }
        }
      }
    }
    _tmp1 = _buffer1;
    for (int _a = 0; _a < 6; ++_a) {
      _tmp1[1*_a] = c10 * _tmp0[1*_a];
    }
    _tmp2 = _buffer0;
    memset(_tmp2, 0, 6 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _k = 0; _k < 6; ++_k) {
        _tmp2[1*_k] += _ut_w[1*_q] * _ut_e_0[1*_k + 6*_q] * _ut_nl[1*_q] * _ut_f_q[1*_q];
      }
    }
    _tmp3 = _buffer2;
    for (int _a = 0; _a < 6; ++_a) {
      _tmp3[1*_a] = c20 * _tmp2[1*_a];
    }
    _tmp4 = _buffer0;
    for (int _a = 0; _a < 6; ++_a) {
      _tmp4[1*_a] = _ut_b[1*_a];
    }
    for (int _a = 0; _a < 6; ++_a) {
      _tmp4[1*_a] += _tmp1[1*_a];
    }
    for (int _a = 0; _a < 6; ++_a) {
      _tmp4[1*_a] += _tmp3[1*_a];
    }
    for (int _a = 0; _a < 6; ++_a) {
      _ut_b[1*_a] = _tmp4[1*_a];
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 6; ++_a) {
        double ref = _ut_b[1*_a];
        double diff = ref - b[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testgrad_u() {
    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

    alignas(16) double d_x_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_1[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_d_x_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_1(_ut_d_x_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<1>::create(d_x_1).copyToView(_view__ut_d_x_1);

    alignas(16) double d_xi_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_xi_0[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_d_xi_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_xi_0(_ut_d_xi_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_xi::view<0>::create(d_xi_0).copyToView(_view__ut_d_xi_0);

    alignas(16) double d_xi_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_xi_1[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_d_xi_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_xi_1(_ut_d_xi_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_xi::view<1>::create(d_xi_1).copyToView(_view__ut_d_xi_1);

    alignas(16) double e_q_T[6] ;
    for (int i = 0; i < 6; ++i) {
      e_q_T[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_e_q_T[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_q_T(_ut_e_q_T, {3, 2}, {0, 0}, {3, 2});
    init::e_q_T::view::create(e_q_T).copyToView(_view__ut_e_q_T);

    alignas(16) double em_0[18] ;
    for (int i = 0; i < 18; ++i) {
      em_0[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_em_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_em_0(_ut_em_0, {3, 6}, {0, 0}, {3, 6});
    init::em::view<0>::create(em_0).copyToView(_view__ut_em_0);

    alignas(16) double em_1[18] ;
    for (int i = 0; i < 18; ++i) {
      em_1[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_em_1[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_em_1(_ut_em_1, {3, 6}, {0, 0}, {3, 6});
    init::em::view<1>::create(em_1).copyToView(_view__ut_em_1);

    alignas(16) double g_0[12] ;
    for (int i = 0; i < 12; ++i) {
      g_0[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_g_0[12]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_g_0(_ut_g_0, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
    init::g::view<0>::create(g_0).copyToView(_view__ut_g_0);

    alignas(16) double g_1[12] ;
    for (int i = 0; i < 12; ++i) {
      g_1[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_g_1[12]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_g_1(_ut_g_1, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
    init::g::view<1>::create(g_1).copyToView(_view__ut_g_1);

    alignas(16) double grad_u[4] ;
    for (int i = 0; i < 4; ++i) {
      grad_u[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_grad_u[4]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_grad_u(_ut_grad_u, {2, 2}, {0, 0}, {2, 2});
    init::grad_u::view::create(grad_u).copyToView(_view__ut_grad_u);

    alignas(16) double k_0[6] ;
    for (int i = 0; i < 6; ++i) {
      k_0[i] = static_cast<double>((i + 10) % 512 + 1);
    }
    alignas(16) double _ut_k_0[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_k_0(_ut_k_0, {6}, {0}, {6});
    init::k::view<0>::create(k_0).copyToView(_view__ut_k_0);

    alignas(16) double k_1[6] ;
    for (int i = 0; i < 6; ++i) {
      k_1[i] = static_cast<double>((i + 11) % 512 + 1);
    }
    alignas(16) double _ut_k_1[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_k_1(_ut_k_1, {6}, {0}, {6});
    init::k::view<1>::create(k_1).copyToView(_view__ut_k_1);

    alignas(16) double minv[4] ;
    for (int i = 0; i < 4; ++i) {
      minv[i] = static_cast<double>((i + 12) % 512 + 1);
    }
    alignas(16) double _ut_minv[4]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_minv(_ut_minv, {2, 2}, {0, 0}, {2, 2});
    init::minv::view::create(minv).copyToView(_view__ut_minv);

    alignas(16) double u_0[6] ;
    for (int i = 0; i < 6; ++i) {
      u_0[i] = static_cast<double>((i + 13) % 512 + 1);
    }
    alignas(16) double _ut_u_0[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_u_0(_ut_u_0, {6}, {0}, {6});
    init::u::view<0>::create(u_0).copyToView(_view__ut_u_0);

    alignas(16) double u_1[6] ;
    for (int i = 0; i < 6; ++i) {
      u_1[i] = static_cast<double>((i + 14) % 512 + 1);
    }
    alignas(16) double _ut_u_1[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_u_1(_ut_u_1, {6}, {0}, {6});
    init::u::view<1>::create(u_1).copyToView(_view__ut_u_1);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 15) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::grad_u krnl;
    krnl.d_x(0) = d_x_0;
    krnl.d_x(1) = d_x_1;
    krnl.d_xi(0) = d_xi_0;
    krnl.d_xi(1) = d_xi_1;
    krnl.e_q_T = e_q_T;
    krnl.em(0) = em_0;
    krnl.em(1) = em_1;
    krnl.g(0) = g_0;
    krnl.g(1) = g_1;
    krnl.grad_u = grad_u;
    krnl.k(0) = k_0;
    krnl.k(1) = k_1;
    krnl.minv = minv;
    krnl.u(0) = u_0;
    krnl.u(1) = u_1;
    krnl.w = w;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9;
    alignas(16) double _buffer0[36] ;
    alignas(16) double _buffer1[36] ;
    alignas(16) double _buffer2[6] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 36 * sizeof(double));
    for (int _e = 0; _e < 2; ++_e) {
      for (int _m = 0; _m < 6; ++_m) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _i = 0; _i < 2; ++_i) {
              _tmp0[1*_i + 2*_k + 12*_q] += _ut_k_0[1*_m] * _ut_em_0[1*_q + 3*_m] * _ut_g_0[1*_e + 2*_i + 4*_q] * _ut_d_xi_0[1*_k + 6*_e + 12*_q];
            }
          }
        }
      }
    }
    _tmp1 = _buffer1;
    for (int _q = 0; _q < 3; ++_q) {
      for (int _i = 0; _i < 2; ++_i) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp1[1*_k + 6*_i + 12*_q] = _tmp0[1*_i + 2*_k + 12*_q];
        }
      }
    }
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _ut_d_x_0[1*_a + 6*_b + 12*_c] = _tmp1[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp2 = _buffer0;
    memset(_tmp2, 0, 36 * sizeof(double));
    for (int _e = 0; _e < 2; ++_e) {
      for (int _m = 0; _m < 6; ++_m) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _i = 0; _i < 2; ++_i) {
              _tmp2[1*_i + 2*_k + 12*_q] += _ut_k_1[1*_m] * _ut_em_1[1*_q + 3*_m] * _ut_g_1[1*_e + 2*_i + 4*_q] * _ut_d_xi_1[1*_k + 6*_e + 12*_q];
            }
          }
        }
      }
    }
    _tmp3 = _buffer1;
    for (int _q = 0; _q < 3; ++_q) {
      for (int _i = 0; _i < 2; ++_i) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp3[1*_k + 6*_i + 12*_q] = _tmp2[1*_i + 2*_k + 12*_q];
        }
      }
    }
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _ut_d_x_1[1*_a + 6*_b + 12*_c] = _tmp3[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp4 = _buffer0;
    memset(_tmp4, 0, 6 * sizeof(double));
    for (int _l = 0; _l < 6; ++_l) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          _tmp4[1*_p + 2*_q] += _ut_d_x_0[1*_l + 6*_p + 12*_q] * _ut_u_0[1*_l];
        }
      }
    }
    _tmp5 = _buffer1;
    memset(_tmp5, 0, 6 * sizeof(double));
    for (int _l = 0; _l < 6; ++_l) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          _tmp5[1*_p + 2*_q] += _ut_d_x_1[1*_l + 6*_p + 12*_q] * _ut_u_1[1*_l];
        }
      }
    }
    _tmp6 = _buffer2;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp6[1*_a + 2*_b] = _tmp4[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp6[1*_a + 2*_b] += _tmp5[1*_a + 2*_b];
      }
    }
    _tmp7 = _buffer0;
    memset(_tmp7, 0, 4 * sizeof(double));
    for (int _r = 0; _r < 2; ++_r) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 2; ++_k) {
            _tmp7[1*_k + 2*_p] += _tmp6[1*_p + 2*_q] * _ut_w[1*_q] * _ut_e_q_T[1*_q + 3*_r] * _ut_minv[1*_r + 2*_k];
          }
        }
      }
    }
    _tmp8 = _buffer1;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp8[1*_a + 2*_b] = 0.5 * _tmp7[1*_a + 2*_b];
      }
    }
    _tmp9 = _buffer2;
    for (int _k = 0; _k < 2; ++_k) {
      for (int _p = 0; _p < 2; ++_p) {
        _tmp9[1*_p + 2*_k] = _tmp8[1*_k + 2*_p];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _ut_grad_u[1*_a + 2*_b] = _tmp9[1*_a + 2*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            double ref = _ut_d_x_0[1*_a + 6*_b + 12*_c];
            double diff = ref - d_x_0[1*_a + 6*_b + 12*_c];
            error += diff * diff;
            refNorm += ref * ref;
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            double ref = _ut_d_x_1[1*_a + 6*_b + 12*_c];
            double diff = ref - d_x_1[1*_a + 6*_b + 12*_c];
            error += diff * diff;
            refNorm += ref * ref;
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          double ref = _ut_grad_u[1*_a + 2*_b];
          double diff = ref - grad_u[1*_a + 2*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
};
#endif
