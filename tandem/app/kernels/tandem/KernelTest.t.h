#ifndef TNDM_KERNELTEST_T_H_
#define TNDM_KERNELTEST_T_H_
#include "kernel.h"
#include "init.h"
#include "yateto.h"
#ifndef NDEBUG
long long libxsmm_num_total_flops = 0;
long long pspamm_num_total_flops = 0;
#endif
#include <cxxtest/TestSuite.h>
namespace tndm {
  namespace unit_test {
    class KernelTestSuite;
  } // namespace unit_test
} // namespace tndm
class tndm::unit_test::KernelTestSuite : public CxxTest::TestSuite {
public:
  void testprecomputeVolume() {
    alignas(16) double Ematerial[42] ;
    for (int i = 0; i < 42; ++i) {
      Ematerial[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_Ematerial[42]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_Ematerial(_ut_Ematerial, {7, 6}, {0, 0}, {7, 6});
    init::Ematerial::view::create(Ematerial).copyToView(_view__ut_Ematerial);

    alignas(16) double J[7] ;
    for (int i = 0; i < 7; ++i) {
      J[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_J(_ut_J, {7}, {0}, {7});
    init::J::view::create(J).copyToView(_view__ut_J);

    alignas(16) double W[7] ;
    for (int i = 0; i < 7; ++i) {
      W[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_W[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_W(_ut_W, {7}, {0}, {7});
    init::W::view::create(W).copyToView(_view__ut_W);

    alignas(16) double lam[6] ;
    for (int i = 0; i < 6; ++i) {
      lam[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_lam[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam(_ut_lam, {6}, {0}, {6});
    init::lam::view::create(lam).copyToView(_view__ut_lam);

    alignas(16) double lam_W_J[7] ;
    for (int i = 0; i < 7; ++i) {
      lam_W_J[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_lam_W_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_W_J(_ut_lam_W_J, {7}, {0}, {7});
    init::lam_W_J::view::create(lam_W_J).copyToView(_view__ut_lam_W_J);

    alignas(16) double mu[6] ;
    for (int i = 0; i < 6; ++i) {
      mu[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_mu[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu(_ut_mu, {6}, {0}, {6});
    init::mu::view::create(mu).copyToView(_view__ut_mu);

    alignas(16) double mu_W_J[7] ;
    for (int i = 0; i < 7; ++i) {
      mu_W_J[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_mu_W_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_W_J(_ut_mu_W_J, {7}, {0}, {7});
    init::mu_W_J::view::create(mu_W_J).copyToView(_view__ut_mu_W_J);

    kernel::precomputeVolume krnl;
    krnl.Ematerial = Ematerial;
    krnl.J = J;
    krnl.W = W;
    krnl.lam = lam;
    krnl.lam_W_J = lam_W_J;
    krnl.mu = mu;
    krnl.mu_W_J = mu_W_J;
    krnl.execute();

    double *_tmp0, *_tmp1;
    alignas(16) double _buffer0[7] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 7 * sizeof(double));
    for (int _t = 0; _t < 6; ++_t) {
      for (int _q = 0; _q < 7; ++_q) {
        _tmp0[1*_q] += _ut_Ematerial[1*_q + 7*_t] * _ut_lam[1*_t] * _ut_J[1*_q] * _ut_W[1*_q];
      }
    }
    for (int _a = 0; _a < 7; ++_a) {
      _ut_lam_W_J[1*_a] = _tmp0[1*_a];
    }
    _tmp1 = _buffer0;
    memset(_tmp1, 0, 7 * sizeof(double));
    for (int _t = 0; _t < 6; ++_t) {
      for (int _q = 0; _q < 7; ++_q) {
        _tmp1[1*_q] += _ut_Ematerial[1*_q + 7*_t] * _ut_mu[1*_t] * _ut_J[1*_q] * _ut_W[1*_q];
      }
    }
    for (int _a = 0; _a < 7; ++_a) {
      _ut_mu_W_J[1*_a] = _tmp1[1*_a];
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 7; ++_a) {
        double ref = _ut_lam_W_J[1*_a];
        double diff = ref - lam_W_J[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 7; ++_a) {
        double ref = _ut_mu_W_J[1*_a];
        double diff = ref - mu_W_J[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testD_x() {
    alignas(16) double D_x[84] ;
    for (int i = 0; i < 84; ++i) {
      D_x[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_D_x[84]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_D_x(_ut_D_x, {6, 2, 7}, {0, 0, 0}, {6, 2, 7});
    init::D_x::view::create(D_x).copyToView(_view__ut_D_x);

    alignas(16) double D_xi[84] ;
    for (int i = 0; i < 84; ++i) {
      D_xi[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_D_xi[84]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_D_xi(_ut_D_xi, {6, 2, 7}, {0, 0, 0}, {6, 2, 7});
    init::D_xi::view::create(D_xi).copyToView(_view__ut_D_xi);

    alignas(16) double G[28] ;
    for (int i = 0; i < 28; ++i) {
      G[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_G[28]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_G(_ut_G, {2, 2, 7}, {0, 0, 0}, {2, 2, 7});
    init::G::view::create(G).copyToView(_view__ut_G);

    kernel::D_x krnl;
    krnl.D_x = D_x;
    krnl.D_xi = D_xi;
    krnl.G = G;
    krnl.execute();

    double *_tmp0, *_tmp1;
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
  void testvolumeOp() {
    alignas(16) double D_x[84] ;
    for (int i = 0; i < 84; ++i) {
      D_x[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_D_x[84]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_D_x(_ut_D_x, {6, 2, 7}, {0, 0, 0}, {6, 2, 7});
    init::D_x::view::create(D_x).copyToView(_view__ut_D_x);

    alignas(16) double U[12] ;
    for (int i = 0; i < 12; ++i) {
      U[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_U[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_U(_ut_U, {6, 2}, {0, 0}, {6, 2});
    init::U::view::create(U).copyToView(_view__ut_U);

    alignas(16) double Unew[12] ;
    for (int i = 0; i < 12; ++i) {
      Unew[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_Unew[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_Unew(_ut_Unew, {6, 2}, {0, 0}, {6, 2});
    init::Unew::view::create(Unew).copyToView(_view__ut_Unew);

    alignas(16) double lam_W_J[7] ;
    for (int i = 0; i < 7; ++i) {
      lam_W_J[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_lam_W_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_W_J(_ut_lam_W_J, {7}, {0}, {7});
    init::lam_W_J::view::create(lam_W_J).copyToView(_view__ut_lam_W_J);

    alignas(16) double mu_W_J[7] ;
    for (int i = 0; i < 7; ++i) {
      mu_W_J[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_mu_W_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_W_J(_ut_mu_W_J, {7}, {0}, {7});
    init::mu_W_J::view::create(mu_W_J).copyToView(_view__ut_mu_W_J);

    kernel::volumeOp krnl;
    krnl.D_x = D_x;
    krnl.U = U;
    krnl.Unew = Unew;
    krnl.lam_W_J = lam_W_J;
    krnl.mu_W_J = mu_W_J;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5;
    alignas(16) double _buffer0[12] ;
    alignas(16) double _buffer1[28] ;
    alignas(16) double _buffer2[28] ;
    alignas(16) double _buffer3[28] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 12 * sizeof(double));
    for (int _r = 0; _r < 2; ++_r) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 7; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp0[1*_k + 6*_p] += _ut_lam_W_J[1*_q] * _ut_D_x[1*_l + 6*_r + 12*_q] * _ut_U[1*_l + 6*_r] * _ut_D_x[1*_k + 6*_p + 12*_q];
            }
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 28 * sizeof(double));
    for (int _l = 0; _l < 6; ++_l) {
      for (int _q = 0; _q < 7; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _j = 0; _j < 2; ++_j) {
            _tmp1[1*_j + 2*_p + 4*_q] += _ut_D_x[1*_l + 6*_j + 12*_q] * _ut_U[1*_l + 6*_p];
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 28 * sizeof(double));
    for (int _l = 0; _l < 6; ++_l) {
      for (int _q = 0; _q < 7; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _j = 0; _j < 2; ++_j) {
            _tmp2[1*_j + 2*_p + 4*_q] += _ut_D_x[1*_l + 6*_p + 12*_q] * _ut_U[1*_l + 6*_j];
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _c = 0; _c < 7; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          _tmp3[1*_a + 2*_b + 4*_c] = _tmp1[1*_a + 2*_b + 4*_c];
        }
      }
    }
    for (int _c = 0; _c < 7; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          _tmp3[1*_a + 2*_b + 4*_c] += _tmp2[1*_a + 2*_b + 4*_c];
        }
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 12 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _q = 0; _q < 7; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp4[1*_k + 6*_p] += _ut_mu_W_J[1*_q] * _ut_D_x[1*_k + 6*_j + 12*_q] * _tmp3[1*_j + 2*_p + 4*_q];
          }
        }
      }
    }
    _tmp5 = _buffer2;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp5[1*_a + 6*_b] = _tmp0[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp5[1*_a + 6*_b] += _tmp4[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_Unew[1*_a + 6*_b] = _tmp5[1*_a + 6*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_Unew[1*_a + 6*_b];
          double diff = ref - Unew[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testassembleVolume() {
    alignas(16) double A[144] ;
    for (int i = 0; i < 144; ++i) {
      A[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_A[144]  = {};
    yateto::DenseTensorView<4,double,unsigned> _view__ut_A(_ut_A, {6, 2, 6, 2}, {0, 0, 0, 0}, {6, 2, 6, 2});
    init::A::view::create(A).copyToView(_view__ut_A);

    alignas(16) double D_x[84] ;
    for (int i = 0; i < 84; ++i) {
      D_x[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_D_x[84]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_D_x(_ut_D_x, {6, 2, 7}, {0, 0, 0}, {6, 2, 7});
    init::D_x::view::create(D_x).copyToView(_view__ut_D_x);

    alignas(16) double delta[4]  = {3.0, 0.0, 0.0, 6.0};
    alignas(16) double _ut_delta[4]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_delta(_ut_delta, {2, 2}, {0, 0}, {2, 2});
    init::delta::view::create(delta).copyToView(_view__ut_delta);

    alignas(16) double lam_W_J[7] ;
    for (int i = 0; i < 7; ++i) {
      lam_W_J[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_lam_W_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_W_J(_ut_lam_W_J, {7}, {0}, {7});
    init::lam_W_J::view::create(lam_W_J).copyToView(_view__ut_lam_W_J);

    alignas(16) double mu_W_J[7] ;
    for (int i = 0; i < 7; ++i) {
      mu_W_J[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_mu_W_J[7]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_W_J(_ut_mu_W_J, {7}, {0}, {7});
    init::mu_W_J::view::create(mu_W_J).copyToView(_view__ut_mu_W_J);

    kernel::assembleVolume krnl;
    krnl.A = A;
    krnl.D_x = D_x;
    krnl.delta = delta;
    krnl.lam_W_J = lam_W_J;
    krnl.mu_W_J = mu_W_J;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6;
    alignas(16) double _buffer0[144] ;
    alignas(16) double _buffer1[336] ;
    alignas(16) double _buffer2[336] ;
    alignas(16) double _buffer3[336] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 7; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp0[1*_k + 6*_l + 36*_p + 72*_u] += _ut_lam_W_J[1*_q] * _ut_D_x[1*_l + 6*_u + 12*_q] * _ut_D_x[1*_k + 6*_p + 12*_q];
            }
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 336 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 7; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp1[1*_j + 2*_l + 12*_p + 24*_q + 168*_u] += _ut_D_x[1*_l + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 336 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 7; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp2[1*_j + 2*_l + 12*_p + 24*_q + 168*_u] += _ut_D_x[1*_l + 6*_p + 12*_q] * _ut_delta[1*_j + 2*_u];
            }
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 7; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 168*_e] = _tmp1[1*_a + 2*_b + 12*_c + 24*_d + 168*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 7; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 168*_e] += _tmp2[1*_a + 2*_b + 12*_c + 24*_d + 168*_e];
            }
          }
        }
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 144 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _q = 0; _q < 7; ++_q) {
        for (int _u = 0; _u < 2; ++_u) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _l = 0; _l < 6; ++_l) {
              for (int _k = 0; _k < 6; ++_k) {
                _tmp4[1*_k + 6*_l + 36*_p + 72*_u] += _ut_mu_W_J[1*_q] * _ut_D_x[1*_k + 6*_j + 12*_q] * _tmp3[1*_j + 2*_l + 12*_p + 24*_q + 168*_u];
              }
            }
          }
        }
      }
    }
    _tmp5 = _buffer2;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 36*_c + 72*_d] = _tmp0[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 36*_c + 72*_d] += _tmp4[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp6 = _buffer3;
    for (int _u = 0; _u < 2; ++_u) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp6[1*_k + 6*_p + 12*_l + 72*_u] = _tmp5[1*_k + 6*_l + 36*_p + 72*_u];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 6; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _ut_A[1*_a + 6*_b + 12*_c + 72*_d] = _tmp6[1*_a + 6*_b + 12*_c + 72*_d];
          }
        }
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _d = 0; _d < 2; ++_d) {
        for (int _c = 0; _c < 6; ++_c) {
          for (int _b = 0; _b < 2; ++_b) {
            for (int _a = 0; _a < 6; ++_a) {
              double ref = _ut_A[1*_a + 6*_b + 12*_c + 72*_d];
              double diff = ref - A[1*_a + 6*_b + 12*_c + 72*_d];
              error += diff * diff;
              refNorm += ref * ref;
            }
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testsurfaceOp() {
    double c00 = 2.0;
    double c01 = 3.0;
    double c10 = 4.0;
    double c11 = 5.0;
    double c20 = 6.0;
    double c21 = 7.0;
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

    alignas(16) double e_0[18] ;
    for (int i = 0; i < 18; ++i) {
      e_0[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_e_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_0(_ut_e_0, {6, 3}, {0, 0}, {6, 3});
    init::e::view<0>::create(e_0).copyToView(_view__ut_e_0);

    alignas(16) double e_1[18] ;
    for (int i = 0; i < 18; ++i) {
      e_1[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_e_1[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_1(_ut_e_1, {6, 3}, {0, 0}, {6, 3});
    init::e::view<1>::create(e_1).copyToView(_view__ut_e_1);

    alignas(16) double lam_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_0[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_0(_ut_lam_w_0, {3}, {0}, {3});
    init::lam_w::view<0>::create(lam_w_0).copyToView(_view__ut_lam_w_0);

    alignas(16) double lam_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_1[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_1(_ut_lam_w_1, {3}, {0}, {3});
    init::lam_w::view<1>::create(lam_w_1).copyToView(_view__ut_lam_w_1);

    alignas(16) double mu_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_0[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_0(_ut_mu_w_0, {3}, {0}, {3});
    init::mu_w::view<0>::create(mu_w_0).copyToView(_view__ut_mu_w_0);

    alignas(16) double mu_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_1[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_1(_ut_mu_w_1, {3}, {0}, {3});
    init::mu_w::view<1>::create(mu_w_1).copyToView(_view__ut_mu_w_1);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double traction_avg[6] ;
    for (int i = 0; i < 6; ++i) {
      traction_avg[i] = static_cast<double>((i + 10) % 512 + 1);
    }
    alignas(16) double _ut_traction_avg[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_traction_avg(_ut_traction_avg, {2, 3}, {0, 0}, {2, 3});
    init::traction_avg::view::create(traction_avg).copyToView(_view__ut_traction_avg);

    alignas(16) double u_0[12] ;
    for (int i = 0; i < 12; ++i) {
      u_0[i] = static_cast<double>((i + 11) % 512 + 1);
    }
    alignas(16) double _ut_u_0[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_u_0(_ut_u_0, {6, 2}, {0, 0}, {6, 2});
    init::u::view<0>::create(u_0).copyToView(_view__ut_u_0);

    alignas(16) double u_1[12] ;
    for (int i = 0; i < 12; ++i) {
      u_1[i] = static_cast<double>((i + 12) % 512 + 1);
    }
    alignas(16) double _ut_u_1[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_u_1(_ut_u_1, {6, 2}, {0, 0}, {6, 2});
    init::u::view<1>::create(u_1).copyToView(_view__ut_u_1);

    alignas(16) double u_jump[6] ;
    for (int i = 0; i < 6; ++i) {
      u_jump[i] = static_cast<double>((i + 13) % 512 + 1);
    }
    alignas(16) double _ut_u_jump[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_u_jump(_ut_u_jump, {2, 3}, {0, 0}, {2, 3});
    init::u_jump::view::create(u_jump).copyToView(_view__ut_u_jump);

    alignas(16) double unew_0[12] ;
    for (int i = 0; i < 12; ++i) {
      unew_0[i] = static_cast<double>((i + 14) % 512 + 1);
    }
    alignas(16) double _ut_unew_0[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_unew_0(_ut_unew_0, {6, 2}, {0, 0}, {6, 2});
    init::unew::view<0>::create(unew_0).copyToView(_view__ut_unew_0);

    alignas(16) double unew_1[12] ;
    for (int i = 0; i < 12; ++i) {
      unew_1[i] = static_cast<double>((i + 15) % 512 + 1);
    }
    alignas(16) double _ut_unew_1[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_unew_1(_ut_unew_1, {6, 2}, {0, 0}, {6, 2});
    init::unew::view<1>::create(unew_1).copyToView(_view__ut_unew_1);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 16) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::surfaceOp krnl;
    krnl.c00 = c00;
    krnl.c01 = c01;
    krnl.c10 = c10;
    krnl.c11 = c11;
    krnl.c20 = c20;
    krnl.c21 = c21;
    krnl.d_x(0) = d_x_0;
    krnl.d_x(1) = d_x_1;
    krnl.e(0) = e_0;
    krnl.e(1) = e_1;
    krnl.lam_w(0) = lam_w_0;
    krnl.lam_w(1) = lam_w_1;
    krnl.mu_w(0) = mu_w_0;
    krnl.mu_w(1) = mu_w_1;
    krnl.n = n;
    krnl.nl = nl;
    krnl.traction_avg = traction_avg;
    krnl.u(0) = u_0;
    krnl.u(1) = u_1;
    krnl.u_jump = u_jump;
    krnl.unew(0) = unew_0;
    krnl.unew(1) = unew_1;
    krnl.w = w;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11, *_tmp12, *_tmp13, *_tmp14, *_tmp15, *_tmp16, *_tmp17, *_tmp18, *_tmp19, *_tmp20, *_tmp21, *_tmp22, *_tmp23, *_tmp24, *_tmp25, *_tmp26, *_tmp27, *_tmp28, *_tmp29, *_tmp30, *_tmp31, *_tmp32, *_tmp33, *_tmp34, *_tmp35, *_tmp36, *_tmp37, *_tmp38, *_tmp39;
    alignas(16) double _buffer0[36] ;
    alignas(16) double _buffer1[36] ;
    alignas(16) double _buffer2[36] ;
    alignas(16) double _buffer3[36] ;
    alignas(16) double _buffer4[36] ;
    alignas(16) double _buffer5[12] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 6 * sizeof(double));
    for (int _s = 0; _s < 2; ++_s) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp0[1*_p + 2*_q] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_l + 6*_s + 12*_q] * _ut_u_0[1*_l + 6*_s] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp1[1*_p + 2*_q] += _ut_d_x_0[1*_l + 6*_j + 12*_q] * _ut_u_0[1*_l + 6*_p] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp2[1*_p + 2*_q] += _ut_d_x_0[1*_l + 6*_p + 12*_q] * _ut_u_0[1*_l + 6*_j] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp3[1*_a + 2*_b] = _tmp1[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp3[1*_a + 2*_b] += _tmp2[1*_a + 2*_b];
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 6 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        _tmp4[1*_p + 2*_q] += _ut_mu_w_0[1*_q] * _tmp3[1*_p + 2*_q];
      }
    }
    _tmp5 = _buffer2;
    memset(_tmp5, 0, 6 * sizeof(double));
    for (int _s = 0; _s < 2; ++_s) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp5[1*_p + 2*_q] += _ut_lam_w_1[1*_q] * _ut_d_x_1[1*_l + 6*_s + 12*_q] * _ut_u_1[1*_l + 6*_s] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp6 = _buffer3;
    memset(_tmp6, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp6[1*_p + 2*_q] += _ut_d_x_1[1*_l + 6*_j + 12*_q] * _ut_u_1[1*_l + 6*_p] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp7 = _buffer4;
    memset(_tmp7, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp7[1*_p + 2*_q] += _ut_d_x_1[1*_l + 6*_p + 12*_q] * _ut_u_1[1*_l + 6*_j] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp8 = _buffer5;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp8[1*_a + 2*_b] = _tmp6[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp8[1*_a + 2*_b] += _tmp7[1*_a + 2*_b];
      }
    }
    _tmp9 = _buffer3;
    memset(_tmp9, 0, 6 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        _tmp9[1*_p + 2*_q] += _ut_mu_w_1[1*_q] * _tmp8[1*_p + 2*_q];
      }
    }
    _tmp10 = _buffer4;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp10[1*_a + 2*_b] = _tmp0[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp10[1*_a + 2*_b] += _tmp4[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp10[1*_a + 2*_b] += _tmp5[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp10[1*_a + 2*_b] += _tmp9[1*_a + 2*_b];
      }
    }
    _tmp11 = _buffer5;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp11[1*_a + 2*_b] = 0.5 * _tmp10[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _ut_traction_avg[1*_a + 2*_b] = _tmp11[1*_a + 2*_b];
      }
    }
    _tmp12 = _buffer0;
    memset(_tmp12, 0, 6 * sizeof(double));
    for (int _l = 0; _l < 6; ++_l) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          _tmp12[1*_p + 2*_q] += _ut_e_0[1*_l + 6*_q] * _ut_u_0[1*_l + 6*_p];
        }
      }
    }
    _tmp13 = _buffer1;
    memset(_tmp13, 0, 6 * sizeof(double));
    for (int _l = 0; _l < 6; ++_l) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          _tmp13[1*_p + 2*_q] += _ut_e_1[1*_l + 6*_q] * _ut_u_1[1*_l + 6*_p];
        }
      }
    }
    _tmp14 = _buffer2;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp14[1*_a + 2*_b] = -1.0 * _tmp13[1*_a + 2*_b];
      }
    }
    _tmp15 = _buffer3;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp15[1*_a + 2*_b] = _tmp12[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp15[1*_a + 2*_b] += _tmp14[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _ut_u_jump[1*_a + 2*_b] = _tmp15[1*_a + 2*_b];
      }
    }
    _tmp16 = _buffer4;
    memset(_tmp16, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp16[1*_k + 6*_p] += _ut_traction_avg[1*_p + 2*_q] * _ut_e_0[1*_k + 6*_q];
        }
      }
    }
    _tmp17 = _buffer5;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp17[1*_a + 6*_b] = c00 * _tmp16[1*_a + 6*_b];
      }
    }
    _tmp18 = _buffer1;
    memset(_tmp18, 0, 12 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp18[1*_k + 6*_p] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_k + 6*_p + 12*_q] * _ut_u_jump[1*_i + 2*_q] * _ut_n[1*_i + 2*_q];
          }
        }
      }
    }
    _tmp19 = _buffer0;
    memset(_tmp19, 0, 36 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp19[1*_k + 6*_p + 12*_q] += _ut_d_x_0[1*_k + 6*_j + 12*_q] * _ut_u_jump[1*_p + 2*_q] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp20 = _buffer2;
    memset(_tmp20, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp20[1*_k + 6*_p + 12*_q] += _ut_d_x_0[1*_k + 6*_i + 12*_q] * _ut_u_jump[1*_i + 2*_q] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp21 = _buffer3;
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _tmp21[1*_a + 6*_b + 12*_c] = _tmp19[1*_a + 6*_b + 12*_c];
        }
      }
    }
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _tmp21[1*_a + 6*_b + 12*_c] += _tmp20[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp22 = _buffer4;
    memset(_tmp22, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp22[1*_k + 6*_p] += _ut_mu_w_0[1*_q] * _tmp21[1*_k + 6*_p + 12*_q];
        }
      }
    }
    _tmp23 = _buffer0;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp23[1*_a + 6*_b] = _tmp18[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp23[1*_a + 6*_b] += _tmp22[1*_a + 6*_b];
      }
    }
    _tmp24 = _buffer2;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp24[1*_a + 6*_b] = c10 * _tmp23[1*_a + 6*_b];
      }
    }
    _tmp25 = _buffer3;
    memset(_tmp25, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp25[1*_k + 6*_p] += _ut_w[1*_q] * _ut_e_0[1*_k + 6*_q] * _ut_u_jump[1*_p + 2*_q] * _ut_nl[1*_q];
        }
      }
    }
    _tmp26 = _buffer1;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp26[1*_a + 6*_b] = c20 * _tmp25[1*_a + 6*_b];
      }
    }
    _tmp27 = _buffer4;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp27[1*_a + 6*_b] = _ut_unew_0[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp27[1*_a + 6*_b] += _tmp17[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp27[1*_a + 6*_b] += _tmp24[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp27[1*_a + 6*_b] += _tmp26[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_unew_0[1*_a + 6*_b] = _tmp27[1*_a + 6*_b];
      }
    }
    _tmp28 = _buffer0;
    memset(_tmp28, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp28[1*_k + 6*_p] += _ut_traction_avg[1*_p + 2*_q] * _ut_e_1[1*_k + 6*_q];
        }
      }
    }
    _tmp29 = _buffer3;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp29[1*_a + 6*_b] = c01 * _tmp28[1*_a + 6*_b];
      }
    }
    _tmp30 = _buffer5;
    memset(_tmp30, 0, 12 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp30[1*_k + 6*_p] += _ut_lam_w_1[1*_q] * _ut_d_x_1[1*_k + 6*_p + 12*_q] * _ut_u_jump[1*_i + 2*_q] * _ut_n[1*_i + 2*_q];
          }
        }
      }
    }
    _tmp31 = _buffer2;
    memset(_tmp31, 0, 36 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp31[1*_k + 6*_p + 12*_q] += _ut_d_x_1[1*_k + 6*_j + 12*_q] * _ut_u_jump[1*_p + 2*_q] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp32 = _buffer1;
    memset(_tmp32, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp32[1*_k + 6*_p + 12*_q] += _ut_d_x_1[1*_k + 6*_i + 12*_q] * _ut_u_jump[1*_i + 2*_q] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp33 = _buffer4;
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _tmp33[1*_a + 6*_b + 12*_c] = _tmp31[1*_a + 6*_b + 12*_c];
        }
      }
    }
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _tmp33[1*_a + 6*_b + 12*_c] += _tmp32[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp34 = _buffer0;
    memset(_tmp34, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp34[1*_k + 6*_p] += _ut_mu_w_1[1*_q] * _tmp33[1*_k + 6*_p + 12*_q];
        }
      }
    }
    _tmp35 = _buffer2;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp35[1*_a + 6*_b] = _tmp30[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp35[1*_a + 6*_b] += _tmp34[1*_a + 6*_b];
      }
    }
    _tmp36 = _buffer1;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp36[1*_a + 6*_b] = c11 * _tmp35[1*_a + 6*_b];
      }
    }
    _tmp37 = _buffer4;
    memset(_tmp37, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp37[1*_k + 6*_p] += _ut_w[1*_q] * _ut_e_1[1*_k + 6*_q] * _ut_u_jump[1*_p + 2*_q] * _ut_nl[1*_q];
        }
      }
    }
    _tmp38 = _buffer5;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp38[1*_a + 6*_b] = c21 * _tmp37[1*_a + 6*_b];
      }
    }
    _tmp39 = _buffer0;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp39[1*_a + 6*_b] = _ut_unew_1[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp39[1*_a + 6*_b] += _tmp29[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp39[1*_a + 6*_b] += _tmp36[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp39[1*_a + 6*_b] += _tmp38[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_unew_1[1*_a + 6*_b] = _tmp39[1*_a + 6*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 3; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          double ref = _ut_traction_avg[1*_a + 2*_b];
          double diff = ref - traction_avg[1*_a + 2*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 3; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          double ref = _ut_u_jump[1*_a + 2*_b];
          double diff = ref - u_jump[1*_a + 2*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_unew_0[1*_a + 6*_b];
          double diff = ref - unew_0[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_unew_1[1*_a + 6*_b];
          double diff = ref - unew_1[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testsurfaceOpBnd() {
    double c00 = 2.0;
    double c10 = 3.0;
    double c20 = 4.0;
    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

    alignas(16) double e_0[18] ;
    for (int i = 0; i < 18; ++i) {
      e_0[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_e_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_0(_ut_e_0, {6, 3}, {0, 0}, {6, 3});
    init::e::view<0>::create(e_0).copyToView(_view__ut_e_0);

    alignas(16) double lam_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_0[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_0(_ut_lam_w_0, {3}, {0}, {3});
    init::lam_w::view<0>::create(lam_w_0).copyToView(_view__ut_lam_w_0);

    alignas(16) double mu_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_0[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_0(_ut_mu_w_0, {3}, {0}, {3});
    init::mu_w::view<0>::create(mu_w_0).copyToView(_view__ut_mu_w_0);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double traction_avg[6] ;
    for (int i = 0; i < 6; ++i) {
      traction_avg[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_traction_avg[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_traction_avg(_ut_traction_avg, {2, 3}, {0, 0}, {2, 3});
    init::traction_avg::view::create(traction_avg).copyToView(_view__ut_traction_avg);

    alignas(16) double u_0[12] ;
    for (int i = 0; i < 12; ++i) {
      u_0[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_u_0[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_u_0(_ut_u_0, {6, 2}, {0, 0}, {6, 2});
    init::u::view<0>::create(u_0).copyToView(_view__ut_u_0);

    alignas(16) double u_jump[6] ;
    for (int i = 0; i < 6; ++i) {
      u_jump[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_u_jump[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_u_jump(_ut_u_jump, {2, 3}, {0, 0}, {2, 3});
    init::u_jump::view::create(u_jump).copyToView(_view__ut_u_jump);

    alignas(16) double unew_0[12] ;
    for (int i = 0; i < 12; ++i) {
      unew_0[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_unew_0[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_unew_0(_ut_unew_0, {6, 2}, {0, 0}, {6, 2});
    init::unew::view<0>::create(unew_0).copyToView(_view__ut_unew_0);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 10) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::surfaceOpBnd krnl;
    krnl.c00 = c00;
    krnl.c10 = c10;
    krnl.c20 = c20;
    krnl.d_x(0) = d_x_0;
    krnl.e(0) = e_0;
    krnl.lam_w(0) = lam_w_0;
    krnl.mu_w(0) = mu_w_0;
    krnl.n = n;
    krnl.nl = nl;
    krnl.traction_avg = traction_avg;
    krnl.u(0) = u_0;
    krnl.u_jump = u_jump;
    krnl.unew(0) = unew_0;
    krnl.w = w;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11, *_tmp12, *_tmp13, *_tmp14, *_tmp15, *_tmp16, *_tmp17, *_tmp18;
    alignas(16) double _buffer0[36] ;
    alignas(16) double _buffer1[12] ;
    alignas(16) double _buffer2[12] ;
    alignas(16) double _buffer3[36] ;
    alignas(16) double _buffer4[36] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 6 * sizeof(double));
    for (int _s = 0; _s < 2; ++_s) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp0[1*_p + 2*_q] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_l + 6*_s + 12*_q] * _ut_u_0[1*_l + 6*_s] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp1[1*_p + 2*_q] += _ut_d_x_0[1*_l + 6*_j + 12*_q] * _ut_u_0[1*_l + 6*_p] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp2[1*_p + 2*_q] += _ut_d_x_0[1*_l + 6*_p + 12*_q] * _ut_u_0[1*_l + 6*_j] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp3[1*_a + 2*_b] = _tmp1[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp3[1*_a + 2*_b] += _tmp2[1*_a + 2*_b];
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 6 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        _tmp4[1*_p + 2*_q] += _ut_mu_w_0[1*_q] * _tmp3[1*_p + 2*_q];
      }
    }
    _tmp5 = _buffer2;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp5[1*_a + 2*_b] = _tmp0[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp5[1*_a + 2*_b] += _tmp4[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _ut_traction_avg[1*_a + 2*_b] = _tmp5[1*_a + 2*_b];
      }
    }
    _tmp6 = _buffer3;
    memset(_tmp6, 0, 6 * sizeof(double));
    for (int _l = 0; _l < 6; ++_l) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          _tmp6[1*_p + 2*_q] += _ut_e_0[1*_l + 6*_q] * _ut_u_0[1*_l + 6*_p];
        }
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _ut_u_jump[1*_a + 2*_b] = _tmp6[1*_a + 2*_b];
      }
    }
    _tmp7 = _buffer0;
    memset(_tmp7, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp7[1*_k + 6*_p] += _ut_traction_avg[1*_p + 2*_q] * _ut_e_0[1*_k + 6*_q];
        }
      }
    }
    _tmp8 = _buffer1;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp8[1*_a + 6*_b] = c00 * _tmp7[1*_a + 6*_b];
      }
    }
    _tmp9 = _buffer2;
    memset(_tmp9, 0, 12 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp9[1*_k + 6*_p] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_k + 6*_p + 12*_q] * _ut_u_jump[1*_i + 2*_q] * _ut_n[1*_i + 2*_q];
          }
        }
      }
    }
    _tmp10 = _buffer3;
    memset(_tmp10, 0, 36 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp10[1*_k + 6*_p + 12*_q] += _ut_d_x_0[1*_k + 6*_j + 12*_q] * _ut_u_jump[1*_p + 2*_q] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp11 = _buffer0;
    memset(_tmp11, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp11[1*_k + 6*_p + 12*_q] += _ut_d_x_0[1*_k + 6*_i + 12*_q] * _ut_u_jump[1*_i + 2*_q] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp12 = _buffer4;
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _tmp12[1*_a + 6*_b + 12*_c] = _tmp10[1*_a + 6*_b + 12*_c];
        }
      }
    }
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _tmp12[1*_a + 6*_b + 12*_c] += _tmp11[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp13 = _buffer3;
    memset(_tmp13, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp13[1*_k + 6*_p] += _ut_mu_w_0[1*_q] * _tmp12[1*_k + 6*_p + 12*_q];
        }
      }
    }
    _tmp14 = _buffer0;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp14[1*_a + 6*_b] = _tmp9[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp14[1*_a + 6*_b] += _tmp13[1*_a + 6*_b];
      }
    }
    _tmp15 = _buffer4;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp15[1*_a + 6*_b] = c10 * _tmp14[1*_a + 6*_b];
      }
    }
    _tmp16 = _buffer2;
    memset(_tmp16, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp16[1*_k + 6*_p] += _ut_w[1*_q] * _ut_e_0[1*_k + 6*_q] * _ut_u_jump[1*_p + 2*_q] * _ut_nl[1*_q];
        }
      }
    }
    _tmp17 = _buffer3;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp17[1*_a + 6*_b] = c20 * _tmp16[1*_a + 6*_b];
      }
    }
    _tmp18 = _buffer0;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp18[1*_a + 6*_b] = _ut_unew_0[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp18[1*_a + 6*_b] += _tmp8[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp18[1*_a + 6*_b] += _tmp15[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp18[1*_a + 6*_b] += _tmp17[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_unew_0[1*_a + 6*_b] = _tmp18[1*_a + 6*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 3; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          double ref = _ut_traction_avg[1*_a + 2*_b];
          double diff = ref - traction_avg[1*_a + 2*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 3; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          double ref = _ut_u_jump[1*_a + 2*_b];
          double diff = ref - u_jump[1*_a + 2*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_unew_0[1*_a + 6*_b];
          double diff = ref - unew_0[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
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

    alignas(16) double F[14] ;
    for (int i = 0; i < 14; ++i) {
      F[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_F[14]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_F(_ut_F, {2, 7}, {0, 0}, {2, 7});
    init::F::view::create(F).copyToView(_view__ut_F);

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

    alignas(16) double b[12] ;
    for (int i = 0; i < 12; ++i) {
      b[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_b[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_b(_ut_b, {6, 2}, {0, 0}, {6, 2});
    init::b::view::create(b).copyToView(_view__ut_b);

    kernel::rhsVolume krnl;
    krnl.E = E;
    krnl.F = F;
    krnl.J = J;
    krnl.W = W;
    krnl.b = b;
    krnl.execute();

    double *_tmp0;
    alignas(16) double _buffer0[12] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 7; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp0[1*_k + 6*_p] += _ut_J[1*_q] * _ut_W[1*_q] * _ut_E[1*_k + 6*_q] * _ut_F[1*_p + 2*_q];
        }
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_b[1*_a + 6*_b] = _tmp0[1*_a + 6*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_b[1*_a + 6*_b];
          double diff = ref - b[1*_a + 6*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void testrhsFacet() {
    double c10 = 2.0;
    double c20 = 3.0;
    alignas(16) double b[12] ;
    for (int i = 0; i < 12; ++i) {
      b[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_b[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_b(_ut_b, {6, 2}, {0, 0}, {6, 2});
    init::b::view::create(b).copyToView(_view__ut_b);

    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

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

    alignas(16) double f[6] ;
    for (int i = 0; i < 6; ++i) {
      f[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_f[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_f(_ut_f, {2, 3}, {0, 0}, {2, 3});
    init::f::view::create(f).copyToView(_view__ut_f);

    alignas(16) double g_0[12] ;
    for (int i = 0; i < 12; ++i) {
      g_0[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_g_0[12]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_g_0(_ut_g_0, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
    init::g::view<0>::create(g_0).copyToView(_view__ut_g_0);

    alignas(16) double lam_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_0[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_0(_ut_lam_w_0, {3}, {0}, {3});
    init::lam_w::view<0>::create(lam_w_0).copyToView(_view__ut_lam_w_0);

    alignas(16) double mu_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_0[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_0(_ut_mu_w_0, {3}, {0}, {3});
    init::mu_w::view<0>::create(mu_w_0).copyToView(_view__ut_mu_w_0);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 10) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::rhsFacet krnl;
    krnl.c10 = c10;
    krnl.c20 = c20;
    krnl.b = b;
    krnl.d_x(0) = d_x_0;
    krnl.d_xi(0) = d_xi_0;
    krnl.e(0) = e_0;
    krnl.f = f;
    krnl.g(0) = g_0;
    krnl.lam_w(0) = lam_w_0;
    krnl.mu_w(0) = mu_w_0;
    krnl.n = n;
    krnl.nl = nl;
    krnl.w = w;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11;
    alignas(16) double _buffer0[36] ;
    alignas(16) double _buffer1[36] ;
    alignas(16) double _buffer2[36] ;
    alignas(16) double _buffer3[36] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 36 * sizeof(double));
    for (int _e = 0; _e < 2; ++_e) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _k = 0; _k < 6; ++_k) {
          for (int _i = 0; _i < 2; ++_i) {
            _tmp0[1*_i + 2*_k + 12*_q] += _ut_g_0[1*_e + 2*_i + 4*_q] * _ut_d_xi_0[1*_k + 6*_e + 12*_q];
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
    memset(_tmp2, 0, 12 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp2[1*_k + 6*_p] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_k + 6*_p + 12*_q] * _ut_f[1*_i + 2*_q] * _ut_n[1*_i + 2*_q];
          }
        }
      }
    }
    _tmp3 = _buffer1;
    memset(_tmp3, 0, 36 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp3[1*_k + 6*_p + 12*_q] += _ut_d_x_0[1*_k + 6*_j + 12*_q] * _ut_f[1*_p + 2*_q] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp4 = _buffer2;
    memset(_tmp4, 0, 36 * sizeof(double));
    for (int _i = 0; _i < 2; ++_i) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp4[1*_k + 6*_p + 12*_q] += _ut_d_x_0[1*_k + 6*_i + 12*_q] * _ut_f[1*_i + 2*_q] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp5 = _buffer3;
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _tmp5[1*_a + 6*_b + 12*_c] = _tmp3[1*_a + 6*_b + 12*_c];
        }
      }
    }
    for (int _c = 0; _c < 3; ++_c) {
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          _tmp5[1*_a + 6*_b + 12*_c] += _tmp4[1*_a + 6*_b + 12*_c];
        }
      }
    }
    _tmp6 = _buffer1;
    memset(_tmp6, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp6[1*_k + 6*_p] += _ut_mu_w_0[1*_q] * _tmp5[1*_k + 6*_p + 12*_q];
        }
      }
    }
    _tmp7 = _buffer2;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp7[1*_a + 6*_b] = _tmp2[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp7[1*_a + 6*_b] += _tmp6[1*_a + 6*_b];
      }
    }
    _tmp8 = _buffer3;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp8[1*_a + 6*_b] = c10 * _tmp7[1*_a + 6*_b];
      }
    }
    _tmp9 = _buffer0;
    memset(_tmp9, 0, 12 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        for (int _k = 0; _k < 6; ++_k) {
          _tmp9[1*_k + 6*_p] += _ut_w[1*_q] * _ut_e_0[1*_k + 6*_q] * _ut_f[1*_p + 2*_q] * _ut_nl[1*_q];
        }
      }
    }
    _tmp10 = _buffer1;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp10[1*_a + 6*_b] = c20 * _tmp9[1*_a + 6*_b];
      }
    }
    _tmp11 = _buffer2;
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp11[1*_a + 6*_b] = _tmp8[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _tmp11[1*_a + 6*_b] += _tmp10[1*_a + 6*_b];
      }
    }
    for (int _b = 0; _b < 2; ++_b) {
      for (int _a = 0; _a < 6; ++_a) {
        _ut_b[1*_a + 6*_b] = _tmp11[1*_a + 6*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 2; ++_b) {
        for (int _a = 0; _a < 6; ++_a) {
          double ref = _ut_b[1*_a + 6*_b];
          double diff = ref - b[1*_a + 6*_b];
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
  void testtraction_avg_proj() {
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

    alignas(16) double enodal[9] ;
    for (int i = 0; i < 9; ++i) {
      enodal[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_enodal[9]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_enodal(_ut_enodal, {3, 3}, {0, 0}, {3, 3});
    init::enodal::view::create(enodal).copyToView(_view__ut_enodal);

    alignas(16) double lam_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_0[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_0(_ut_lam_w_0, {3}, {0}, {3});
    init::lam_w::view<0>::create(lam_w_0).copyToView(_view__ut_lam_w_0);

    alignas(16) double lam_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_1[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_1(_ut_lam_w_1, {3}, {0}, {3});
    init::lam_w::view<1>::create(lam_w_1).copyToView(_view__ut_lam_w_1);

    alignas(16) double minv[9] ;
    for (int i = 0; i < 9; ++i) {
      minv[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_minv[9]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_minv(_ut_minv, {3, 3}, {0, 0}, {3, 3});
    init::minv::view::create(minv).copyToView(_view__ut_minv);

    alignas(16) double mu_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_0[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_0(_ut_mu_w_0, {3}, {0}, {3});
    init::mu_w::view<0>::create(mu_w_0).copyToView(_view__ut_mu_w_0);

    alignas(16) double mu_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_1[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_1(_ut_mu_w_1, {3}, {0}, {3});
    init::mu_w::view<1>::create(mu_w_1).copyToView(_view__ut_mu_w_1);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double traction_avg[6] ;
    for (int i = 0; i < 6; ++i) {
      traction_avg[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_traction_avg[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_traction_avg(_ut_traction_avg, {2, 3}, {0, 0}, {2, 3});
    init::traction_avg::view::create(traction_avg).copyToView(_view__ut_traction_avg);

    alignas(16) double traction_avg_proj[6] ;
    for (int i = 0; i < 6; ++i) {
      traction_avg_proj[i] = static_cast<double>((i + 10) % 512 + 1);
    }
    alignas(16) double _ut_traction_avg_proj[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_traction_avg_proj(_ut_traction_avg_proj, {2, 3}, {0, 0}, {2, 3});
    init::traction_avg_proj::view::create(traction_avg_proj).copyToView(_view__ut_traction_avg_proj);

    alignas(16) double u_0[12] ;
    for (int i = 0; i < 12; ++i) {
      u_0[i] = static_cast<double>((i + 11) % 512 + 1);
    }
    alignas(16) double _ut_u_0[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_u_0(_ut_u_0, {6, 2}, {0, 0}, {6, 2});
    init::u::view<0>::create(u_0).copyToView(_view__ut_u_0);

    alignas(16) double u_1[12] ;
    for (int i = 0; i < 12; ++i) {
      u_1[i] = static_cast<double>((i + 12) % 512 + 1);
    }
    alignas(16) double _ut_u_1[12]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_u_1(_ut_u_1, {6, 2}, {0, 0}, {6, 2});
    init::u::view<1>::create(u_1).copyToView(_view__ut_u_1);

    kernel::traction_avg_proj krnl;
    krnl.d_x(0) = d_x_0;
    krnl.d_x(1) = d_x_1;
    krnl.enodal = enodal;
    krnl.lam_w(0) = lam_w_0;
    krnl.lam_w(1) = lam_w_1;
    krnl.minv = minv;
    krnl.mu_w(0) = mu_w_0;
    krnl.mu_w(1) = mu_w_1;
    krnl.n = n;
    krnl.traction_avg = traction_avg;
    krnl.traction_avg_proj = traction_avg_proj;
    krnl.u(0) = u_0;
    krnl.u(1) = u_1;
    krnl.execute();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11, *_tmp12, *_tmp13;
    alignas(16) double _buffer0[6] ;
    alignas(16) double _buffer1[6] ;
    alignas(16) double _buffer2[6] ;
    alignas(16) double _buffer3[6] ;
    alignas(16) double _buffer4[6] ;
    alignas(16) double _buffer5[6] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 6 * sizeof(double));
    for (int _s = 0; _s < 2; ++_s) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp0[1*_p + 2*_q] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_l + 6*_s + 12*_q] * _ut_u_0[1*_l + 6*_s] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp1[1*_p + 2*_q] += _ut_d_x_0[1*_l + 6*_j + 12*_q] * _ut_u_0[1*_l + 6*_p] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp2[1*_p + 2*_q] += _ut_d_x_0[1*_l + 6*_p + 12*_q] * _ut_u_0[1*_l + 6*_j] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp3[1*_a + 2*_b] = _tmp1[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp3[1*_a + 2*_b] += _tmp2[1*_a + 2*_b];
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 6 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        _tmp4[1*_p + 2*_q] += _ut_mu_w_0[1*_q] * _tmp3[1*_p + 2*_q];
      }
    }
    _tmp5 = _buffer2;
    memset(_tmp5, 0, 6 * sizeof(double));
    for (int _s = 0; _s < 2; ++_s) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp5[1*_p + 2*_q] += _ut_lam_w_1[1*_q] * _ut_d_x_1[1*_l + 6*_s + 12*_q] * _ut_u_1[1*_l + 6*_s] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp6 = _buffer3;
    memset(_tmp6, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp6[1*_p + 2*_q] += _ut_d_x_1[1*_l + 6*_j + 12*_q] * _ut_u_1[1*_l + 6*_p] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp7 = _buffer4;
    memset(_tmp7, 0, 6 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            _tmp7[1*_p + 2*_q] += _ut_d_x_1[1*_l + 6*_p + 12*_q] * _ut_u_1[1*_l + 6*_j] * _ut_n[1*_j + 2*_q];
          }
        }
      }
    }
    _tmp8 = _buffer5;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp8[1*_a + 2*_b] = _tmp6[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp8[1*_a + 2*_b] += _tmp7[1*_a + 2*_b];
      }
    }
    _tmp9 = _buffer3;
    memset(_tmp9, 0, 6 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _p = 0; _p < 2; ++_p) {
        _tmp9[1*_p + 2*_q] += _ut_mu_w_1[1*_q] * _tmp8[1*_p + 2*_q];
      }
    }
    _tmp10 = _buffer4;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp10[1*_a + 2*_b] = _tmp0[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp10[1*_a + 2*_b] += _tmp4[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp10[1*_a + 2*_b] += _tmp5[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp10[1*_a + 2*_b] += _tmp9[1*_a + 2*_b];
      }
    }
    _tmp11 = _buffer5;
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _tmp11[1*_a + 2*_b] = 0.5 * _tmp10[1*_a + 2*_b];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _ut_traction_avg[1*_a + 2*_b] = _tmp11[1*_a + 2*_b];
      }
    }
    _tmp12 = _buffer0;
    memset(_tmp12, 0, 6 * sizeof(double));
    for (int _l = 0; _l < 3; ++_l) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 3; ++_k) {
            _tmp12[1*_k + 3*_p] += _ut_traction_avg[1*_p + 2*_q] * _ut_enodal[1*_q + 3*_l] * _ut_minv[1*_l + 3*_k];
          }
        }
      }
    }
    _tmp13 = _buffer1;
    for (int _k = 0; _k < 3; ++_k) {
      for (int _p = 0; _p < 2; ++_p) {
        _tmp13[1*_p + 2*_k] = _tmp12[1*_k + 3*_p];
      }
    }
    for (int _b = 0; _b < 3; ++_b) {
      for (int _a = 0; _a < 2; ++_a) {
        _ut_traction_avg_proj[1*_a + 2*_b] = _tmp13[1*_a + 2*_b];
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 3; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          double ref = _ut_traction_avg[1*_a + 2*_b];
          double diff = ref - traction_avg[1*_a + 2*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _b = 0; _b < 3; ++_b) {
        for (int _a = 0; _a < 2; ++_a) {
          double ref = _ut_traction_avg_proj[1*_a + 2*_b];
          double diff = ref - traction_avg_proj[1*_a + 2*_b];
          error += diff * diff;
          refNorm += ref * ref;
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void test_precomputeSurface_0() {
    alignas(16) double ematerial_0[18] ;
    for (int i = 0; i < 18; ++i) {
      ematerial_0[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_ematerial_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_ematerial_0(_ut_ematerial_0, {3, 6}, {0, 0}, {3, 6});
    init::ematerial::view<0>::create(ematerial_0).copyToView(_view__ut_ematerial_0);

    alignas(16) double lam[6] ;
    for (int i = 0; i < 6; ++i) {
      lam[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_lam[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam(_ut_lam, {6}, {0}, {6});
    init::lam::view::create(lam).copyToView(_view__ut_lam);

    alignas(16) double lam_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_0[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_0(_ut_lam_w_0, {3}, {0}, {3});
    init::lam_w::view<0>::create(lam_w_0).copyToView(_view__ut_lam_w_0);

    alignas(16) double mu[6] ;
    for (int i = 0; i < 6; ++i) {
      mu[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_mu[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu(_ut_mu, {6}, {0}, {6});
    init::mu::view::create(mu).copyToView(_view__ut_mu);

    alignas(16) double mu_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_0[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_0(_ut_mu_w_0, {3}, {0}, {3});
    init::mu_w::view<0>::create(mu_w_0).copyToView(_view__ut_mu_w_0);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::precomputeSurface krnl;
    krnl.ematerial(0) = ematerial_0;
    krnl.lam = lam;
    krnl.lam_w(0) = lam_w_0;
    krnl.mu = mu;
    krnl.mu_w(0) = mu_w_0;
    krnl.w = w;
    krnl.execute0();

    double *_tmp0, *_tmp1;
    alignas(16) double _buffer0[3] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 3 * sizeof(double));
    for (int _t = 0; _t < 6; ++_t) {
      for (int _q = 0; _q < 3; ++_q) {
        _tmp0[1*_q] += _ut_ematerial_0[1*_q + 3*_t] * _ut_lam[1*_t] * _ut_w[1*_q];
      }
    }
    for (int _a = 0; _a < 3; ++_a) {
      _ut_lam_w_0[1*_a] = _tmp0[1*_a];
    }
    _tmp1 = _buffer0;
    memset(_tmp1, 0, 3 * sizeof(double));
    for (int _t = 0; _t < 6; ++_t) {
      for (int _q = 0; _q < 3; ++_q) {
        _tmp1[1*_q] += _ut_ematerial_0[1*_q + 3*_t] * _ut_mu[1*_t] * _ut_w[1*_q];
      }
    }
    for (int _a = 0; _a < 3; ++_a) {
      _ut_mu_w_0[1*_a] = _tmp1[1*_a];
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 3; ++_a) {
        double ref = _ut_lam_w_0[1*_a];
        double diff = ref - lam_w_0[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 3; ++_a) {
        double ref = _ut_mu_w_0[1*_a];
        double diff = ref - mu_w_0[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void test_precomputeSurface_1() {
    alignas(16) double ematerial_1[18] ;
    for (int i = 0; i < 18; ++i) {
      ematerial_1[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_ematerial_1[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_ematerial_1(_ut_ematerial_1, {3, 6}, {0, 0}, {3, 6});
    init::ematerial::view<1>::create(ematerial_1).copyToView(_view__ut_ematerial_1);

    alignas(16) double lam[6] ;
    for (int i = 0; i < 6; ++i) {
      lam[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_lam[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam(_ut_lam, {6}, {0}, {6});
    init::lam::view::create(lam).copyToView(_view__ut_lam);

    alignas(16) double lam_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_1[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_1(_ut_lam_w_1, {3}, {0}, {3});
    init::lam_w::view<1>::create(lam_w_1).copyToView(_view__ut_lam_w_1);

    alignas(16) double mu[6] ;
    for (int i = 0; i < 6; ++i) {
      mu[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_mu[6]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu(_ut_mu, {6}, {0}, {6});
    init::mu::view::create(mu).copyToView(_view__ut_mu);

    alignas(16) double mu_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_1[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_1(_ut_mu_w_1, {3}, {0}, {3});
    init::mu_w::view<1>::create(mu_w_1).copyToView(_view__ut_mu_w_1);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::precomputeSurface krnl;
    krnl.ematerial(1) = ematerial_1;
    krnl.lam = lam;
    krnl.lam_w(1) = lam_w_1;
    krnl.mu = mu;
    krnl.mu_w(1) = mu_w_1;
    krnl.w = w;
    krnl.execute1();

    double *_tmp0, *_tmp1;
    alignas(16) double _buffer0[3] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 3 * sizeof(double));
    for (int _t = 0; _t < 6; ++_t) {
      for (int _q = 0; _q < 3; ++_q) {
        _tmp0[1*_q] += _ut_ematerial_1[1*_q + 3*_t] * _ut_lam[1*_t] * _ut_w[1*_q];
      }
    }
    for (int _a = 0; _a < 3; ++_a) {
      _ut_lam_w_1[1*_a] = _tmp0[1*_a];
    }
    _tmp1 = _buffer0;
    memset(_tmp1, 0, 3 * sizeof(double));
    for (int _t = 0; _t < 6; ++_t) {
      for (int _q = 0; _q < 3; ++_q) {
        _tmp1[1*_q] += _ut_ematerial_1[1*_q + 3*_t] * _ut_mu[1*_t] * _ut_w[1*_q];
      }
    }
    for (int _a = 0; _a < 3; ++_a) {
      _ut_mu_w_1[1*_a] = _tmp1[1*_a];
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 3; ++_a) {
        double ref = _ut_lam_w_1[1*_a];
        double diff = ref - lam_w_1[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _a = 0; _a < 3; ++_a) {
        double ref = _ut_mu_w_1[1*_a];
        double diff = ref - mu_w_1[1*_a];
        error += diff * diff;
        refNorm += ref * ref;
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void test_d_x_0() {
    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

    alignas(16) double d_xi_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_xi_0[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_d_xi_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_xi_0(_ut_d_xi_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_xi::view<0>::create(d_xi_0).copyToView(_view__ut_d_xi_0);

    alignas(16) double g_0[12] ;
    for (int i = 0; i < 12; ++i) {
      g_0[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_g_0[12]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_g_0(_ut_g_0, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
    init::g::view<0>::create(g_0).copyToView(_view__ut_g_0);

    kernel::d_x krnl;
    krnl.d_x(0) = d_x_0;
    krnl.d_xi(0) = d_xi_0;
    krnl.g(0) = g_0;
    krnl.execute0();

    double *_tmp0, *_tmp1;
    alignas(16) double _buffer0[36] ;
    alignas(16) double _buffer1[36] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 36 * sizeof(double));
    for (int _e = 0; _e < 2; ++_e) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _k = 0; _k < 6; ++_k) {
          for (int _i = 0; _i < 2; ++_i) {
            _tmp0[1*_i + 2*_k + 12*_q] += _ut_g_0[1*_e + 2*_i + 4*_q] * _ut_d_xi_0[1*_k + 6*_e + 12*_q];
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
  void test_d_x_1() {
    alignas(16) double d_x_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_1[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_d_x_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_1(_ut_d_x_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<1>::create(d_x_1).copyToView(_view__ut_d_x_1);

    alignas(16) double d_xi_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_xi_1[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_d_xi_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_xi_1(_ut_d_xi_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_xi::view<1>::create(d_xi_1).copyToView(_view__ut_d_xi_1);

    alignas(16) double g_1[12] ;
    for (int i = 0; i < 12; ++i) {
      g_1[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_g_1[12]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_g_1(_ut_g_1, {2, 2, 3}, {0, 0, 0}, {2, 2, 3});
    init::g::view<1>::create(g_1).copyToView(_view__ut_g_1);

    kernel::d_x krnl;
    krnl.d_x(1) = d_x_1;
    krnl.d_xi(1) = d_xi_1;
    krnl.g(1) = g_1;
    krnl.execute1();

    double *_tmp0, *_tmp1;
    alignas(16) double _buffer0[36] ;
    alignas(16) double _buffer1[36] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 36 * sizeof(double));
    for (int _e = 0; _e < 2; ++_e) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _k = 0; _k < 6; ++_k) {
          for (int _i = 0; _i < 2; ++_i) {
            _tmp0[1*_i + 2*_k + 12*_q] += _ut_g_1[1*_e + 2*_i + 4*_q] * _ut_d_xi_1[1*_k + 6*_e + 12*_q];
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
  void test_assembleSurface_0() {
    double c00 = 2.0;
    double c10 = 3.0;
    double c20 = 4.0;
    alignas(16) double a_0_0[144] ;
    for (int i = 0; i < 144; ++i) {
      a_0_0[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_a_0_0[144]  = {};
    yateto::DenseTensorView<4,double,unsigned> _view__ut_a_0_0(_ut_a_0_0, {6, 2, 6, 2}, {0, 0, 0, 0}, {6, 2, 6, 2});
    init::a::view<0,0>::create(a_0_0).copyToView(_view__ut_a_0_0);

    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

    alignas(16) double delta[4]  = {3.0, 0.0, 0.0, 6.0};
    alignas(16) double _ut_delta[4]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_delta(_ut_delta, {2, 2}, {0, 0}, {2, 2});
    init::delta::view::create(delta).copyToView(_view__ut_delta);

    alignas(16) double e_0[18] ;
    for (int i = 0; i < 18; ++i) {
      e_0[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_e_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_0(_ut_e_0, {6, 3}, {0, 0}, {6, 3});
    init::e::view<0>::create(e_0).copyToView(_view__ut_e_0);

    alignas(16) double lam_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_0[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_0(_ut_lam_w_0, {3}, {0}, {3});
    init::lam_w::view<0>::create(lam_w_0).copyToView(_view__ut_lam_w_0);

    alignas(16) double mu_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_0[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_0(_ut_mu_w_0, {3}, {0}, {3});
    init::mu_w::view<0>::create(mu_w_0).copyToView(_view__ut_mu_w_0);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::assembleSurface krnl;
    krnl.c00 = c00;
    krnl.c10 = c10;
    krnl.c20 = c20;
    krnl.a(0,0) = a_0_0;
    krnl.d_x(0) = d_x_0;
    krnl.delta = delta;
    krnl.e(0) = e_0;
    krnl.lam_w(0) = lam_w_0;
    krnl.mu_w(0) = mu_w_0;
    krnl.n = n;
    krnl.nl = nl;
    krnl.w = w;
    krnl.execute0();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11, *_tmp12, *_tmp13, *_tmp14, *_tmp15, *_tmp16, *_tmp17, *_tmp18, *_tmp19;
    alignas(16) double _buffer0[144] ;
    alignas(16) double _buffer1[144] ;
    alignas(16) double _buffer2[144] ;
    alignas(16) double _buffer3[144] ;
    alignas(16) double _buffer4[144] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 72 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            _tmp0[1*_l + 6*_p + 12*_q + 36*_u] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_l + 6*_u + 12*_q] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp1[1*_j + 2*_l + 12*_p + 24*_q + 72*_u] += _ut_d_x_0[1*_l + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp2[1*_j + 2*_l + 12*_p + 24*_q + 72*_u] += _ut_d_x_0[1*_l + 6*_p + 12*_q] * _ut_delta[1*_j + 2*_u];
            }
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] = _tmp1[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] += _tmp2[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 72 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _l = 0; _l < 6; ++_l) {
              _tmp4[1*_l + 6*_p + 12*_q + 36*_u] += _ut_mu_w_0[1*_q] * _ut_n[1*_j + 2*_q] * _tmp3[1*_j + 2*_l + 12*_p + 24*_q + 72*_u];
            }
          }
        }
      }
    }
    _tmp5 = _buffer2;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 12*_c + 36*_d] = _tmp0[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 12*_c + 36*_d] += _tmp4[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    _tmp6 = _buffer3;
    memset(_tmp6, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp6[1*_k + 6*_l + 36*_p + 72*_u] += _ut_e_0[1*_k + 6*_q] * _tmp5[1*_l + 6*_p + 12*_q + 36*_u];
            }
          }
        }
      }
    }
    _tmp7 = _buffer0;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp7[1*_a + 6*_b + 36*_c + 72*_d] = c00 * _tmp6[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp8 = _buffer1;
    memset(_tmp8, 0, 72 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp8[1*_k + 6*_p + 12*_q + 36*_u] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_k + 6*_p + 12*_q] * _ut_n[1*_u + 2*_q];
          }
        }
      }
    }
    _tmp9 = _buffer2;
    memset(_tmp9, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp9[1*_j + 2*_k + 12*_p + 24*_q + 72*_u] += _ut_d_x_0[1*_k + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp10 = _buffer3;
    memset(_tmp10, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp10[1*_j + 2*_k + 12*_p + 24*_q + 72*_u] += _ut_d_x_0[1*_k + 6*_u + 12*_q] * _ut_delta[1*_j + 2*_p];
            }
          }
        }
      }
    }
    _tmp11 = _buffer4;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp11[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] = _tmp9[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp11[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] += _tmp10[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    _tmp12 = _buffer2;
    memset(_tmp12, 0, 72 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp12[1*_k + 6*_p + 12*_q + 36*_u] += _ut_mu_w_0[1*_q] * _ut_n[1*_j + 2*_q] * _tmp11[1*_j + 2*_k + 12*_p + 24*_q + 72*_u];
            }
          }
        }
      }
    }
    _tmp13 = _buffer3;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp13[1*_a + 6*_b + 12*_c + 36*_d] = _tmp8[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp13[1*_a + 6*_b + 12*_c + 36*_d] += _tmp12[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    _tmp14 = _buffer4;
    memset(_tmp14, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp14[1*_k + 6*_l + 36*_p + 72*_u] += _ut_e_0[1*_l + 6*_q] * _tmp13[1*_k + 6*_p + 12*_q + 36*_u];
            }
          }
        }
      }
    }
    _tmp15 = _buffer1;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp15[1*_a + 6*_b + 36*_c + 72*_d] = c10 * _tmp14[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp16 = _buffer2;
    memset(_tmp16, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp16[1*_k + 6*_l + 36*_p + 72*_u] += _ut_delta[1*_p + 2*_u] * _ut_e_0[1*_l + 6*_q] * _ut_e_0[1*_k + 6*_q] * _ut_w[1*_q] * _ut_nl[1*_q];
            }
          }
        }
      }
    }
    _tmp17 = _buffer3;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp17[1*_a + 6*_b + 36*_c + 72*_d] = c20 * _tmp16[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp18 = _buffer4;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] = _tmp7[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] += _tmp15[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] += _tmp17[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp19 = _buffer2;
    for (int _u = 0; _u < 2; ++_u) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp19[1*_k + 6*_p + 12*_l + 72*_u] = _tmp18[1*_k + 6*_l + 36*_p + 72*_u];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 6; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _ut_a_0_0[1*_a + 6*_b + 12*_c + 72*_d] = _tmp19[1*_a + 6*_b + 12*_c + 72*_d];
          }
        }
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _d = 0; _d < 2; ++_d) {
        for (int _c = 0; _c < 6; ++_c) {
          for (int _b = 0; _b < 2; ++_b) {
            for (int _a = 0; _a < 6; ++_a) {
              double ref = _ut_a_0_0[1*_a + 6*_b + 12*_c + 72*_d];
              double diff = ref - a_0_0[1*_a + 6*_b + 12*_c + 72*_d];
              error += diff * diff;
              refNorm += ref * ref;
            }
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void test_assembleSurface_2() {
    double c00 = 2.0;
    double c11 = 3.0;
    double c21 = 4.0;
    alignas(16) double a_0_1[144] ;
    for (int i = 0; i < 144; ++i) {
      a_0_1[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_a_0_1[144]  = {};
    yateto::DenseTensorView<4,double,unsigned> _view__ut_a_0_1(_ut_a_0_1, {6, 2, 6, 2}, {0, 0, 0, 0}, {6, 2, 6, 2});
    init::a::view<0,1>::create(a_0_1).copyToView(_view__ut_a_0_1);

    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

    alignas(16) double d_x_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_1[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_d_x_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_1(_ut_d_x_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<1>::create(d_x_1).copyToView(_view__ut_d_x_1);

    alignas(16) double delta[4]  = {4.0, 0.0, 0.0, 7.0};
    alignas(16) double _ut_delta[4]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_delta(_ut_delta, {2, 2}, {0, 0}, {2, 2});
    init::delta::view::create(delta).copyToView(_view__ut_delta);

    alignas(16) double e_0[18] ;
    for (int i = 0; i < 18; ++i) {
      e_0[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_e_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_0(_ut_e_0, {6, 3}, {0, 0}, {6, 3});
    init::e::view<0>::create(e_0).copyToView(_view__ut_e_0);

    alignas(16) double e_1[18] ;
    for (int i = 0; i < 18; ++i) {
      e_1[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_e_1[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_1(_ut_e_1, {6, 3}, {0, 0}, {6, 3});
    init::e::view<1>::create(e_1).copyToView(_view__ut_e_1);

    alignas(16) double lam_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_0[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_0(_ut_lam_w_0, {3}, {0}, {3});
    init::lam_w::view<0>::create(lam_w_0).copyToView(_view__ut_lam_w_0);

    alignas(16) double lam_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_1[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_1(_ut_lam_w_1, {3}, {0}, {3});
    init::lam_w::view<1>::create(lam_w_1).copyToView(_view__ut_lam_w_1);

    alignas(16) double mu_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_0[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_0(_ut_mu_w_0, {3}, {0}, {3});
    init::mu_w::view<0>::create(mu_w_0).copyToView(_view__ut_mu_w_0);

    alignas(16) double mu_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_1[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_1(_ut_mu_w_1, {3}, {0}, {3});
    init::mu_w::view<1>::create(mu_w_1).copyToView(_view__ut_mu_w_1);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 10) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 11) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 12) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::assembleSurface krnl;
    krnl.c00 = c00;
    krnl.c11 = c11;
    krnl.c21 = c21;
    krnl.a(0,1) = a_0_1;
    krnl.d_x(0) = d_x_0;
    krnl.d_x(1) = d_x_1;
    krnl.delta = delta;
    krnl.e(0) = e_0;
    krnl.e(1) = e_1;
    krnl.lam_w(0) = lam_w_0;
    krnl.lam_w(1) = lam_w_1;
    krnl.mu_w(0) = mu_w_0;
    krnl.mu_w(1) = mu_w_1;
    krnl.n = n;
    krnl.nl = nl;
    krnl.w = w;
    krnl.execute2();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11, *_tmp12, *_tmp13, *_tmp14, *_tmp15, *_tmp16, *_tmp17, *_tmp18, *_tmp19;
    alignas(16) double _buffer0[144] ;
    alignas(16) double _buffer1[144] ;
    alignas(16) double _buffer2[144] ;
    alignas(16) double _buffer3[144] ;
    alignas(16) double _buffer4[144] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 72 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            _tmp0[1*_l + 6*_p + 12*_q + 36*_u] += _ut_lam_w_1[1*_q] * _ut_d_x_1[1*_l + 6*_u + 12*_q] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp1[1*_j + 2*_l + 12*_p + 24*_q + 72*_u] += _ut_d_x_1[1*_l + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp2[1*_j + 2*_l + 12*_p + 24*_q + 72*_u] += _ut_d_x_1[1*_l + 6*_p + 12*_q] * _ut_delta[1*_j + 2*_u];
            }
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] = _tmp1[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] += _tmp2[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 72 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _l = 0; _l < 6; ++_l) {
              _tmp4[1*_l + 6*_p + 12*_q + 36*_u] += _ut_mu_w_1[1*_q] * _ut_n[1*_j + 2*_q] * _tmp3[1*_j + 2*_l + 12*_p + 24*_q + 72*_u];
            }
          }
        }
      }
    }
    _tmp5 = _buffer2;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 12*_c + 36*_d] = _tmp0[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 12*_c + 36*_d] += _tmp4[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    _tmp6 = _buffer3;
    memset(_tmp6, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp6[1*_k + 6*_l + 36*_p + 72*_u] += _ut_e_0[1*_k + 6*_q] * _tmp5[1*_l + 6*_p + 12*_q + 36*_u];
            }
          }
        }
      }
    }
    _tmp7 = _buffer0;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp7[1*_a + 6*_b + 36*_c + 72*_d] = c00 * _tmp6[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp8 = _buffer1;
    memset(_tmp8, 0, 72 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp8[1*_k + 6*_p + 12*_q + 36*_u] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_k + 6*_p + 12*_q] * _ut_n[1*_u + 2*_q];
          }
        }
      }
    }
    _tmp9 = _buffer2;
    memset(_tmp9, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp9[1*_j + 2*_k + 12*_p + 24*_q + 72*_u] += _ut_d_x_0[1*_k + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp10 = _buffer3;
    memset(_tmp10, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp10[1*_j + 2*_k + 12*_p + 24*_q + 72*_u] += _ut_d_x_0[1*_k + 6*_u + 12*_q] * _ut_delta[1*_j + 2*_p];
            }
          }
        }
      }
    }
    _tmp11 = _buffer4;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp11[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] = _tmp9[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp11[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] += _tmp10[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    _tmp12 = _buffer2;
    memset(_tmp12, 0, 72 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp12[1*_k + 6*_p + 12*_q + 36*_u] += _ut_mu_w_0[1*_q] * _ut_n[1*_j + 2*_q] * _tmp11[1*_j + 2*_k + 12*_p + 24*_q + 72*_u];
            }
          }
        }
      }
    }
    _tmp13 = _buffer3;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp13[1*_a + 6*_b + 12*_c + 36*_d] = _tmp8[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp13[1*_a + 6*_b + 12*_c + 36*_d] += _tmp12[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    _tmp14 = _buffer4;
    memset(_tmp14, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp14[1*_k + 6*_l + 36*_p + 72*_u] += _ut_e_1[1*_l + 6*_q] * _tmp13[1*_k + 6*_p + 12*_q + 36*_u];
            }
          }
        }
      }
    }
    _tmp15 = _buffer1;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp15[1*_a + 6*_b + 36*_c + 72*_d] = c11 * _tmp14[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp16 = _buffer2;
    memset(_tmp16, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp16[1*_k + 6*_l + 36*_p + 72*_u] += _ut_delta[1*_p + 2*_u] * _ut_e_1[1*_l + 6*_q] * _ut_e_0[1*_k + 6*_q] * _ut_w[1*_q] * _ut_nl[1*_q];
            }
          }
        }
      }
    }
    _tmp17 = _buffer3;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp17[1*_a + 6*_b + 36*_c + 72*_d] = c21 * _tmp16[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp18 = _buffer4;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] = _tmp7[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] += _tmp15[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] += _tmp17[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp19 = _buffer2;
    for (int _u = 0; _u < 2; ++_u) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp19[1*_k + 6*_p + 12*_l + 72*_u] = _tmp18[1*_k + 6*_l + 36*_p + 72*_u];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 6; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _ut_a_0_1[1*_a + 6*_b + 12*_c + 72*_d] = _tmp19[1*_a + 6*_b + 12*_c + 72*_d];
          }
        }
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _d = 0; _d < 2; ++_d) {
        for (int _c = 0; _c < 6; ++_c) {
          for (int _b = 0; _b < 2; ++_b) {
            for (int _a = 0; _a < 6; ++_a) {
              double ref = _ut_a_0_1[1*_a + 6*_b + 12*_c + 72*_d];
              double diff = ref - a_0_1[1*_a + 6*_b + 12*_c + 72*_d];
              error += diff * diff;
              refNorm += ref * ref;
            }
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void test_assembleSurface_1() {
    double c01 = 2.0;
    double c10 = 3.0;
    double c21 = 4.0;
    alignas(16) double a_1_0[144] ;
    for (int i = 0; i < 144; ++i) {
      a_1_0[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_a_1_0[144]  = {};
    yateto::DenseTensorView<4,double,unsigned> _view__ut_a_1_0(_ut_a_1_0, {6, 2, 6, 2}, {0, 0, 0, 0}, {6, 2, 6, 2});
    init::a::view<1,0>::create(a_1_0).copyToView(_view__ut_a_1_0);

    alignas(16) double d_x_0[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_0[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_d_x_0[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_0(_ut_d_x_0, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<0>::create(d_x_0).copyToView(_view__ut_d_x_0);

    alignas(16) double d_x_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_1[i] = static_cast<double>((i + 2) % 512 + 1);
    }
    alignas(16) double _ut_d_x_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_1(_ut_d_x_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<1>::create(d_x_1).copyToView(_view__ut_d_x_1);

    alignas(16) double delta[4]  = {4.0, 0.0, 0.0, 7.0};
    alignas(16) double _ut_delta[4]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_delta(_ut_delta, {2, 2}, {0, 0}, {2, 2});
    init::delta::view::create(delta).copyToView(_view__ut_delta);

    alignas(16) double e_0[18] ;
    for (int i = 0; i < 18; ++i) {
      e_0[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_e_0[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_0(_ut_e_0, {6, 3}, {0, 0}, {6, 3});
    init::e::view<0>::create(e_0).copyToView(_view__ut_e_0);

    alignas(16) double e_1[18] ;
    for (int i = 0; i < 18; ++i) {
      e_1[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_e_1[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_1(_ut_e_1, {6, 3}, {0, 0}, {6, 3});
    init::e::view<1>::create(e_1).copyToView(_view__ut_e_1);

    alignas(16) double lam_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_0[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_0(_ut_lam_w_0, {3}, {0}, {3});
    init::lam_w::view<0>::create(lam_w_0).copyToView(_view__ut_lam_w_0);

    alignas(16) double lam_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_1[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_1(_ut_lam_w_1, {3}, {0}, {3});
    init::lam_w::view<1>::create(lam_w_1).copyToView(_view__ut_lam_w_1);

    alignas(16) double mu_w_0[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_0[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_0[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_0(_ut_mu_w_0, {3}, {0}, {3});
    init::mu_w::view<0>::create(mu_w_0).copyToView(_view__ut_mu_w_0);

    alignas(16) double mu_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_1[i] = static_cast<double>((i + 9) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_1(_ut_mu_w_1, {3}, {0}, {3});
    init::mu_w::view<1>::create(mu_w_1).copyToView(_view__ut_mu_w_1);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 10) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 11) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 12) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::assembleSurface krnl;
    krnl.c01 = c01;
    krnl.c10 = c10;
    krnl.c21 = c21;
    krnl.a(1,0) = a_1_0;
    krnl.d_x(0) = d_x_0;
    krnl.d_x(1) = d_x_1;
    krnl.delta = delta;
    krnl.e(0) = e_0;
    krnl.e(1) = e_1;
    krnl.lam_w(0) = lam_w_0;
    krnl.lam_w(1) = lam_w_1;
    krnl.mu_w(0) = mu_w_0;
    krnl.mu_w(1) = mu_w_1;
    krnl.n = n;
    krnl.nl = nl;
    krnl.w = w;
    krnl.execute1();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11, *_tmp12, *_tmp13, *_tmp14, *_tmp15, *_tmp16, *_tmp17, *_tmp18, *_tmp19;
    alignas(16) double _buffer0[144] ;
    alignas(16) double _buffer1[144] ;
    alignas(16) double _buffer2[144] ;
    alignas(16) double _buffer3[144] ;
    alignas(16) double _buffer4[144] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 72 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            _tmp0[1*_l + 6*_p + 12*_q + 36*_u] += _ut_lam_w_0[1*_q] * _ut_d_x_0[1*_l + 6*_u + 12*_q] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp1[1*_j + 2*_l + 12*_p + 24*_q + 72*_u] += _ut_d_x_0[1*_l + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp2[1*_j + 2*_l + 12*_p + 24*_q + 72*_u] += _ut_d_x_0[1*_l + 6*_p + 12*_q] * _ut_delta[1*_j + 2*_u];
            }
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] = _tmp1[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] += _tmp2[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 72 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _l = 0; _l < 6; ++_l) {
              _tmp4[1*_l + 6*_p + 12*_q + 36*_u] += _ut_mu_w_0[1*_q] * _ut_n[1*_j + 2*_q] * _tmp3[1*_j + 2*_l + 12*_p + 24*_q + 72*_u];
            }
          }
        }
      }
    }
    _tmp5 = _buffer2;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 12*_c + 36*_d] = _tmp0[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 12*_c + 36*_d] += _tmp4[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    _tmp6 = _buffer3;
    memset(_tmp6, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp6[1*_k + 6*_l + 36*_p + 72*_u] += _ut_e_1[1*_k + 6*_q] * _tmp5[1*_l + 6*_p + 12*_q + 36*_u];
            }
          }
        }
      }
    }
    _tmp7 = _buffer0;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp7[1*_a + 6*_b + 36*_c + 72*_d] = c01 * _tmp6[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp8 = _buffer1;
    memset(_tmp8, 0, 72 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp8[1*_k + 6*_p + 12*_q + 36*_u] += _ut_lam_w_1[1*_q] * _ut_d_x_1[1*_k + 6*_p + 12*_q] * _ut_n[1*_u + 2*_q];
          }
        }
      }
    }
    _tmp9 = _buffer2;
    memset(_tmp9, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp9[1*_j + 2*_k + 12*_p + 24*_q + 72*_u] += _ut_d_x_1[1*_k + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp10 = _buffer3;
    memset(_tmp10, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp10[1*_j + 2*_k + 12*_p + 24*_q + 72*_u] += _ut_d_x_1[1*_k + 6*_u + 12*_q] * _ut_delta[1*_j + 2*_p];
            }
          }
        }
      }
    }
    _tmp11 = _buffer4;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp11[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] = _tmp9[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp11[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] += _tmp10[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    _tmp12 = _buffer2;
    memset(_tmp12, 0, 72 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp12[1*_k + 6*_p + 12*_q + 36*_u] += _ut_mu_w_1[1*_q] * _ut_n[1*_j + 2*_q] * _tmp11[1*_j + 2*_k + 12*_p + 24*_q + 72*_u];
            }
          }
        }
      }
    }
    _tmp13 = _buffer3;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp13[1*_a + 6*_b + 12*_c + 36*_d] = _tmp8[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp13[1*_a + 6*_b + 12*_c + 36*_d] += _tmp12[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    _tmp14 = _buffer4;
    memset(_tmp14, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp14[1*_k + 6*_l + 36*_p + 72*_u] += _ut_e_0[1*_l + 6*_q] * _tmp13[1*_k + 6*_p + 12*_q + 36*_u];
            }
          }
        }
      }
    }
    _tmp15 = _buffer1;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp15[1*_a + 6*_b + 36*_c + 72*_d] = c10 * _tmp14[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp16 = _buffer2;
    memset(_tmp16, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp16[1*_k + 6*_l + 36*_p + 72*_u] += _ut_delta[1*_p + 2*_u] * _ut_e_0[1*_l + 6*_q] * _ut_e_1[1*_k + 6*_q] * _ut_w[1*_q] * _ut_nl[1*_q];
            }
          }
        }
      }
    }
    _tmp17 = _buffer3;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp17[1*_a + 6*_b + 36*_c + 72*_d] = c21 * _tmp16[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp18 = _buffer4;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] = _tmp7[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] += _tmp15[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] += _tmp17[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp19 = _buffer2;
    for (int _u = 0; _u < 2; ++_u) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp19[1*_k + 6*_p + 12*_l + 72*_u] = _tmp18[1*_k + 6*_l + 36*_p + 72*_u];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 6; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _ut_a_1_0[1*_a + 6*_b + 12*_c + 72*_d] = _tmp19[1*_a + 6*_b + 12*_c + 72*_d];
          }
        }
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _d = 0; _d < 2; ++_d) {
        for (int _c = 0; _c < 6; ++_c) {
          for (int _b = 0; _b < 2; ++_b) {
            for (int _a = 0; _a < 6; ++_a) {
              double ref = _ut_a_1_0[1*_a + 6*_b + 12*_c + 72*_d];
              double diff = ref - a_1_0[1*_a + 6*_b + 12*_c + 72*_d];
              error += diff * diff;
              refNorm += ref * ref;
            }
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
  void test_assembleSurface_3() {
    double c01 = 2.0;
    double c11 = 3.0;
    double c20 = 4.0;
    alignas(16) double a_1_1[144] ;
    for (int i = 0; i < 144; ++i) {
      a_1_1[i] = static_cast<double>((i + 0) % 512 + 1);
    }
    alignas(16) double _ut_a_1_1[144]  = {};
    yateto::DenseTensorView<4,double,unsigned> _view__ut_a_1_1(_ut_a_1_1, {6, 2, 6, 2}, {0, 0, 0, 0}, {6, 2, 6, 2});
    init::a::view<1,1>::create(a_1_1).copyToView(_view__ut_a_1_1);

    alignas(16) double d_x_1[36] ;
    for (int i = 0; i < 36; ++i) {
      d_x_1[i] = static_cast<double>((i + 1) % 512 + 1);
    }
    alignas(16) double _ut_d_x_1[36]  = {};
    yateto::DenseTensorView<3,double,unsigned> _view__ut_d_x_1(_ut_d_x_1, {6, 2, 3}, {0, 0, 0}, {6, 2, 3});
    init::d_x::view<1>::create(d_x_1).copyToView(_view__ut_d_x_1);

    alignas(16) double delta[4]  = {3.0, 0.0, 0.0, 6.0};
    alignas(16) double _ut_delta[4]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_delta(_ut_delta, {2, 2}, {0, 0}, {2, 2});
    init::delta::view::create(delta).copyToView(_view__ut_delta);

    alignas(16) double e_1[18] ;
    for (int i = 0; i < 18; ++i) {
      e_1[i] = static_cast<double>((i + 3) % 512 + 1);
    }
    alignas(16) double _ut_e_1[18]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_e_1(_ut_e_1, {6, 3}, {0, 0}, {6, 3});
    init::e::view<1>::create(e_1).copyToView(_view__ut_e_1);

    alignas(16) double lam_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      lam_w_1[i] = static_cast<double>((i + 4) % 512 + 1);
    }
    alignas(16) double _ut_lam_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_lam_w_1(_ut_lam_w_1, {3}, {0}, {3});
    init::lam_w::view<1>::create(lam_w_1).copyToView(_view__ut_lam_w_1);

    alignas(16) double mu_w_1[3] ;
    for (int i = 0; i < 3; ++i) {
      mu_w_1[i] = static_cast<double>((i + 5) % 512 + 1);
    }
    alignas(16) double _ut_mu_w_1[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_mu_w_1(_ut_mu_w_1, {3}, {0}, {3});
    init::mu_w::view<1>::create(mu_w_1).copyToView(_view__ut_mu_w_1);

    alignas(16) double n[6] ;
    for (int i = 0; i < 6; ++i) {
      n[i] = static_cast<double>((i + 6) % 512 + 1);
    }
    alignas(16) double _ut_n[6]  = {};
    yateto::DenseTensorView<2,double,unsigned> _view__ut_n(_ut_n, {2, 3}, {0, 0}, {2, 3});
    init::n::view::create(n).copyToView(_view__ut_n);

    alignas(16) double nl[3] ;
    for (int i = 0; i < 3; ++i) {
      nl[i] = static_cast<double>((i + 7) % 512 + 1);
    }
    alignas(16) double _ut_nl[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_nl(_ut_nl, {3}, {0}, {3});
    init::nl::view::create(nl).copyToView(_view__ut_nl);

    alignas(16) double w[3] ;
    for (int i = 0; i < 3; ++i) {
      w[i] = static_cast<double>((i + 8) % 512 + 1);
    }
    alignas(16) double _ut_w[3]  = {};
    yateto::DenseTensorView<1,double,unsigned> _view__ut_w(_ut_w, {3}, {0}, {3});
    init::w::view::create(w).copyToView(_view__ut_w);

    kernel::assembleSurface krnl;
    krnl.c01 = c01;
    krnl.c11 = c11;
    krnl.c20 = c20;
    krnl.a(1,1) = a_1_1;
    krnl.d_x(1) = d_x_1;
    krnl.delta = delta;
    krnl.e(1) = e_1;
    krnl.lam_w(1) = lam_w_1;
    krnl.mu_w(1) = mu_w_1;
    krnl.n = n;
    krnl.nl = nl;
    krnl.w = w;
    krnl.execute3();

    double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp5, *_tmp6, *_tmp7, *_tmp8, *_tmp9, *_tmp10, *_tmp11, *_tmp12, *_tmp13, *_tmp14, *_tmp15, *_tmp16, *_tmp17, *_tmp18, *_tmp19;
    alignas(16) double _buffer0[144] ;
    alignas(16) double _buffer1[144] ;
    alignas(16) double _buffer2[144] ;
    alignas(16) double _buffer3[144] ;
    alignas(16) double _buffer4[144] ;
    _tmp0 = _buffer0;
    memset(_tmp0, 0, 72 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            _tmp0[1*_l + 6*_p + 12*_q + 36*_u] += _ut_lam_w_1[1*_q] * _ut_d_x_1[1*_l + 6*_u + 12*_q] * _ut_n[1*_p + 2*_q];
          }
        }
      }
    }
    _tmp1 = _buffer1;
    memset(_tmp1, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp1[1*_j + 2*_l + 12*_p + 24*_q + 72*_u] += _ut_d_x_1[1*_l + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp2 = _buffer2;
    memset(_tmp2, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp2[1*_j + 2*_l + 12*_p + 24*_q + 72*_u] += _ut_d_x_1[1*_l + 6*_p + 12*_q] * _ut_delta[1*_j + 2*_u];
            }
          }
        }
      }
    }
    _tmp3 = _buffer3;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] = _tmp1[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp3[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] += _tmp2[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    _tmp4 = _buffer1;
    memset(_tmp4, 0, 72 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _l = 0; _l < 6; ++_l) {
              _tmp4[1*_l + 6*_p + 12*_q + 36*_u] += _ut_mu_w_1[1*_q] * _ut_n[1*_j + 2*_q] * _tmp3[1*_j + 2*_l + 12*_p + 24*_q + 72*_u];
            }
          }
        }
      }
    }
    _tmp5 = _buffer2;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 12*_c + 36*_d] = _tmp0[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp5[1*_a + 6*_b + 12*_c + 36*_d] += _tmp4[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    _tmp6 = _buffer3;
    memset(_tmp6, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp6[1*_k + 6*_l + 36*_p + 72*_u] += _ut_e_1[1*_k + 6*_q] * _tmp5[1*_l + 6*_p + 12*_q + 36*_u];
            }
          }
        }
      }
    }
    _tmp7 = _buffer0;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp7[1*_a + 6*_b + 36*_c + 72*_d] = c01 * _tmp6[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp8 = _buffer1;
    memset(_tmp8, 0, 72 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp8[1*_k + 6*_p + 12*_q + 36*_u] += _ut_lam_w_1[1*_q] * _ut_d_x_1[1*_k + 6*_p + 12*_q] * _ut_n[1*_u + 2*_q];
          }
        }
      }
    }
    _tmp9 = _buffer2;
    memset(_tmp9, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp9[1*_j + 2*_k + 12*_p + 24*_q + 72*_u] += _ut_d_x_1[1*_k + 6*_j + 12*_q] * _ut_delta[1*_p + 2*_u];
            }
          }
        }
      }
    }
    _tmp10 = _buffer3;
    memset(_tmp10, 0, 144 * sizeof(double));
    for (int _u = 0; _u < 2; ++_u) {
      for (int _q = 0; _q < 3; ++_q) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            for (int _j = 0; _j < 2; ++_j) {
              _tmp10[1*_j + 2*_k + 12*_p + 24*_q + 72*_u] += _ut_d_x_1[1*_k + 6*_u + 12*_q] * _ut_delta[1*_j + 2*_p];
            }
          }
        }
      }
    }
    _tmp11 = _buffer4;
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp11[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] = _tmp9[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    for (int _e = 0; _e < 2; ++_e) {
      for (int _d = 0; _d < 3; ++_d) {
        for (int _c = 0; _c < 2; ++_c) {
          for (int _b = 0; _b < 6; ++_b) {
            for (int _a = 0; _a < 2; ++_a) {
              _tmp11[1*_a + 2*_b + 12*_c + 24*_d + 72*_e] += _tmp10[1*_a + 2*_b + 12*_c + 24*_d + 72*_e];
            }
          }
        }
      }
    }
    _tmp12 = _buffer2;
    memset(_tmp12, 0, 72 * sizeof(double));
    for (int _j = 0; _j < 2; ++_j) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _q = 0; _q < 3; ++_q) {
          for (int _p = 0; _p < 2; ++_p) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp12[1*_k + 6*_p + 12*_q + 36*_u] += _ut_mu_w_1[1*_q] * _ut_n[1*_j + 2*_q] * _tmp11[1*_j + 2*_k + 12*_p + 24*_q + 72*_u];
            }
          }
        }
      }
    }
    _tmp13 = _buffer3;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp13[1*_a + 6*_b + 12*_c + 36*_d] = _tmp8[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 3; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp13[1*_a + 6*_b + 12*_c + 36*_d] += _tmp12[1*_a + 6*_b + 12*_c + 36*_d];
          }
        }
      }
    }
    _tmp14 = _buffer4;
    memset(_tmp14, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp14[1*_k + 6*_l + 36*_p + 72*_u] += _ut_e_1[1*_l + 6*_q] * _tmp13[1*_k + 6*_p + 12*_q + 36*_u];
            }
          }
        }
      }
    }
    _tmp15 = _buffer1;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp15[1*_a + 6*_b + 36*_c + 72*_d] = c11 * _tmp14[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp16 = _buffer2;
    memset(_tmp16, 0, 144 * sizeof(double));
    for (int _q = 0; _q < 3; ++_q) {
      for (int _u = 0; _u < 2; ++_u) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _l = 0; _l < 6; ++_l) {
            for (int _k = 0; _k < 6; ++_k) {
              _tmp16[1*_k + 6*_l + 36*_p + 72*_u] += _ut_delta[1*_p + 2*_u] * _ut_e_1[1*_l + 6*_q] * _ut_e_1[1*_k + 6*_q] * _ut_w[1*_q] * _ut_nl[1*_q];
            }
          }
        }
      }
    }
    _tmp17 = _buffer3;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp17[1*_a + 6*_b + 36*_c + 72*_d] = c20 * _tmp16[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp18 = _buffer4;
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] = _tmp7[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] += _tmp15[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 2; ++_c) {
        for (int _b = 0; _b < 6; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _tmp18[1*_a + 6*_b + 36*_c + 72*_d] += _tmp17[1*_a + 6*_b + 36*_c + 72*_d];
          }
        }
      }
    }
    _tmp19 = _buffer2;
    for (int _u = 0; _u < 2; ++_u) {
      for (int _l = 0; _l < 6; ++_l) {
        for (int _p = 0; _p < 2; ++_p) {
          for (int _k = 0; _k < 6; ++_k) {
            _tmp19[1*_k + 6*_p + 12*_l + 72*_u] = _tmp18[1*_k + 6*_l + 36*_p + 72*_u];
          }
        }
      }
    }
    for (int _d = 0; _d < 2; ++_d) {
      for (int _c = 0; _c < 6; ++_c) {
        for (int _b = 0; _b < 2; ++_b) {
          for (int _a = 0; _a < 6; ++_a) {
            _ut_a_1_1[1*_a + 6*_b + 12*_c + 72*_d] = _tmp19[1*_a + 6*_b + 12*_c + 72*_d];
          }
        }
      }
    }
    {
      double error = 0.0;
      double refNorm = 0.0;
      for (int _d = 0; _d < 2; ++_d) {
        for (int _c = 0; _c < 6; ++_c) {
          for (int _b = 0; _b < 2; ++_b) {
            for (int _a = 0; _a < 6; ++_a) {
              double ref = _ut_a_1_1[1*_a + 6*_b + 12*_c + 72*_d];
              double diff = ref - a_1_1[1*_a + 6*_b + 12*_c + 72*_d];
              error += diff * diff;
              refNorm += ref * ref;
            }
          }
        }
      }
      TS_ASSERT_LESS_THAN(sqrt(error/refNorm), 2.22e-14);
    }
  }
};
#endif
