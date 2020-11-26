#include "Eigen/Eigen"
#include <cassert>
#include <cstring>
#include <cstdlib>
#include "subroutine.h"
#include "kernel.h"
namespace tndm {
  namespace poisson {
    constexpr unsigned long const kernel::project_K::NonZeroFlops;
    constexpr unsigned long const kernel::project_K::HardwareFlops;
    void kernel::project_K::execute() {
      assert(Em != nullptr);
      assert(K != nullptr);
      assert(K_Q != nullptr);
      assert(W != nullptr);
      assert(matMinv != nullptr);
      double *_tmp0, *_tmp1;
      alignas(16) double _buffer0[7] ;
      alignas(16) double _buffer1[6] ;
      _tmp0 = _buffer0;
      #pragma omp simd
      for (int _q = 0; _q < 7; ++_q) {
        _tmp0[1*_q] = K_Q[1*_q] * W[1*_q];
      }
      _tmp1 = _buffer1;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,7,6>,Eigen::Unaligned,Stride<7,1>> _mapA(const_cast<double*>(Em));
        Map<Matrix<double,7,1>,Eigen::Unaligned,Stride<7,1>> _mapB(const_cast<double*>(_tmp0));
        Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_tmp1);
        _mapC = _mapA.transpose()*_mapB;
      }
          
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(matMinv));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(_tmp1));
        Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(K);
        _mapC = _mapA*_mapB;
      }
          
    }
    constexpr unsigned long const kernel::assembleVolume::NonZeroFlops;
    constexpr unsigned long const kernel::assembleVolume::HardwareFlops;
    void kernel::assembleVolume::execute() {
      assert(A != nullptr);
      assert(D_x != nullptr);
      assert(D_xi != nullptr);
      assert(Em != nullptr);
      assert(G != nullptr);
      assert(J != nullptr);
      assert(K != nullptr);
      assert(W != nullptr);
      double *_tmp1, *_tmp2, *_tmp3, *_tmp4;
      alignas(16) double _buffer0[84] ;
      alignas(16) double _buffer1[7] ;
      alignas(16) double _buffer2[7] ;
      for (int _q = 0; _q < 7; ++_q) {
        double const* _A = D_xi + 12*_q;
        double const* _B = G + 4*_q;
        double * _C = D_x + 12*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,2>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      _tmp1 = _buffer0;
      #pragma omp simd
      for (int _q = 0; _q < 7; ++_q) {
        _tmp1[1*_q] = J[1*_q] * W[1*_q];
      }
      _tmp2 = _buffer1;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,7,6>,Eigen::Unaligned,Stride<7,1>> _mapA(const_cast<double*>(Em));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(K));
        Map<Matrix<double,7,1>,Eigen::Unaligned,Stride<7,1>> _mapC(_tmp2);
        _mapC = _mapA*_mapB;
      }
          
      _tmp3 = _buffer2;
      #pragma omp simd
      for (int _q = 0; _q < 7; ++_q) {
        _tmp3[1*_q] = _tmp1[1*_q] * _tmp2[1*_q];
      }
      _tmp4 = _buffer0;
      for (int _q = 0; _q < 7; ++_q) {
        for (int _i = 0; _i < 2; ++_i) {
          #pragma omp simd
          for (int _k = 0; _k < 6; ++_k) {
            _tmp4[1*_k + 6*_i + 12*_q] = D_x[1*_k + 6*_i + 12*_q] * _tmp3[1*_q];
          }
        }
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,14>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp4));
        Map<Matrix<double,6,14>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(D_x));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(A);
        _mapC = _mapA*_mapB.transpose();
      }
          
    }
    constexpr unsigned long const kernel::assembleFacetLocal::NonZeroFlops;
    constexpr unsigned long const kernel::assembleFacetLocal::HardwareFlops;
    void kernel::assembleFacetLocal::execute() {
      assert(!std::isnan(c00));
      assert(!std::isnan(c10));
      assert(!std::isnan(c20));
      assert(K != nullptr);
      assert(a(0,0) != nullptr);
      assert(d_x(0) != nullptr);
      assert(d_xi(0) != nullptr);
      assert(e(0) != nullptr);
      assert(em(0) != nullptr);
      assert(g(0) != nullptr);
      assert(n != nullptr);
      assert(nl != nullptr);
      assert(w != nullptr);
      double *_tmp0, *_tmp1, *_tmp3, *_tmp4, *_tmp7, *_tmp8, *_tmp11, *_tmp12;
      alignas(16) double _buffer0[6] ;
      alignas(16) double _buffer1[18] ;
      _tmp0 = _buffer0;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,3,6>,Eigen::Unaligned,Stride<3,1>> _mapA(const_cast<double*>(em(0)));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(K));
        Map<Matrix<double,3,1>,Eigen::Unaligned,Stride<3,1>> _mapC(_tmp0);
        _mapC = _mapA*_mapB;
      }
          
      _tmp1 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        for (int _i = 0; _i < 2; ++_i) {
          #pragma omp simd
          for (int _e = 0; _e < 2; ++_e) {
            _tmp1[1*_e + 2*_i + 4*_q] = g(0)[1*_e + 2*_i + 4*_q] * _tmp0[1*_q];
          }
        }
      }
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_xi(0) + 12*_q;
        double const* _B = _tmp1 + 4*_q;
        double * _C = d_x(0) + 12*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,2>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      _tmp3 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp3[1*_i + 2*_q] = w[1*_q] * n[1*_i + 2*_q];
        }
      }
      _tmp4 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_x(0) + 12*_q;
        double const* _B = _tmp3 + 2*_q;
        double * _C = _tmp4 + 6*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp4));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(e(0)));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(0,0));
        _mapC = c00*_mapA*_mapB.transpose();
      }
          
      _tmp7 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp7[1*_i + 2*_q] = w[1*_q] * n[1*_i + 2*_q];
        }
      }
      _tmp8 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_x(0) + 12*_q;
        double const* _B = _tmp7 + 2*_q;
        double * _C = _tmp8 + 6*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(e(0)));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(_tmp8));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(0,0));
        _mapC.noalias() += c10*_mapA*_mapB.transpose();
      }
          
      _tmp11 = _buffer0;
      #pragma omp simd
      for (int _q = 0; _q < 3; ++_q) {
        _tmp11[1*_q] = w[1*_q] * nl[1*_q];
      }
      _tmp12 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _k = 0; _k < 6; ++_k) {
          _tmp12[1*_k + 6*_q] = e(0)[1*_k + 6*_q] * _tmp11[1*_q];
        }
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp12));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(e(0)));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(0,0));
        _mapC.noalias() += c20*_mapA*_mapB.transpose();
      }
          
    }
    constexpr unsigned long const kernel::assembleFacetNeighbour::NonZeroFlops;
    constexpr unsigned long const kernel::assembleFacetNeighbour::HardwareFlops;
    void kernel::assembleFacetNeighbour::execute() {
      assert(!std::isnan(c00));
      assert(!std::isnan(c01));
      assert(!std::isnan(c10));
      assert(!std::isnan(c11));
      assert(!std::isnan(c20));
      assert(!std::isnan(c21));
      assert(K != nullptr);
      assert(a(0,1) != nullptr);
      assert(a(1,0) != nullptr);
      assert(a(1,1) != nullptr);
      assert(d_x(0) != nullptr);
      assert(d_x(1) != nullptr);
      assert(d_xi(1) != nullptr);
      assert(e(0) != nullptr);
      assert(e(1) != nullptr);
      assert(em(1) != nullptr);
      assert(g(1) != nullptr);
      assert(n != nullptr);
      assert(nl != nullptr);
      assert(w != nullptr);
      double *_tmp0, *_tmp1, *_tmp3, *_tmp4, *_tmp7, *_tmp8, *_tmp11, *_tmp12, *_tmp16, *_tmp17, *_tmp20, *_tmp21, *_tmp24, *_tmp25, *_tmp29, *_tmp30, *_tmp33, *_tmp34, *_tmp37, *_tmp38;
      alignas(16) double _buffer0[6] ;
      alignas(16) double _buffer1[18] ;
      _tmp0 = _buffer0;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,3,6>,Eigen::Unaligned,Stride<3,1>> _mapA(const_cast<double*>(em(1)));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(K));
        Map<Matrix<double,3,1>,Eigen::Unaligned,Stride<3,1>> _mapC(_tmp0);
        _mapC = _mapA*_mapB;
      }
          
      _tmp1 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        for (int _i = 0; _i < 2; ++_i) {
          #pragma omp simd
          for (int _e = 0; _e < 2; ++_e) {
            _tmp1[1*_e + 2*_i + 4*_q] = g(1)[1*_e + 2*_i + 4*_q] * _tmp0[1*_q];
          }
        }
      }
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_xi(1) + 12*_q;
        double const* _B = _tmp1 + 4*_q;
        double * _C = d_x(1) + 12*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,2>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      _tmp3 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp3[1*_i + 2*_q] = w[1*_q] * n[1*_i + 2*_q];
        }
      }
      _tmp4 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_x(0) + 12*_q;
        double const* _B = _tmp3 + 2*_q;
        double * _C = _tmp4 + 6*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp4));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(e(1)));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(0,1));
        _mapC = c01*_mapA*_mapB.transpose();
      }
          
      _tmp7 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp7[1*_i + 2*_q] = w[1*_q] * n[1*_i + 2*_q];
        }
      }
      _tmp8 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_x(1) + 12*_q;
        double const* _B = _tmp7 + 2*_q;
        double * _C = _tmp8 + 6*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(e(0)));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(_tmp8));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(0,1));
        _mapC.noalias() += c10*_mapA*_mapB.transpose();
      }
          
      _tmp11 = _buffer0;
      #pragma omp simd
      for (int _q = 0; _q < 3; ++_q) {
        _tmp11[1*_q] = w[1*_q] * nl[1*_q];
      }
      _tmp12 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _k = 0; _k < 6; ++_k) {
          _tmp12[1*_k + 6*_q] = e(0)[1*_k + 6*_q] * _tmp11[1*_q];
        }
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp12));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(e(1)));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(0,1));
        _mapC.noalias() += c21*_mapA*_mapB.transpose();
      }
          
      _tmp16 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp16[1*_i + 2*_q] = w[1*_q] * n[1*_i + 2*_q];
        }
      }
      _tmp17 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_x(1) + 12*_q;
        double const* _B = _tmp16 + 2*_q;
        double * _C = _tmp17 + 6*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp17));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(e(0)));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(1,0));
        _mapC = c00*_mapA*_mapB.transpose();
      }
          
      _tmp20 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp20[1*_i + 2*_q] = w[1*_q] * n[1*_i + 2*_q];
        }
      }
      _tmp21 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_x(0) + 12*_q;
        double const* _B = _tmp20 + 2*_q;
        double * _C = _tmp21 + 6*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(e(1)));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(_tmp21));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(1,0));
        _mapC.noalias() += c11*_mapA*_mapB.transpose();
      }
          
      _tmp24 = _buffer0;
      #pragma omp simd
      for (int _q = 0; _q < 3; ++_q) {
        _tmp24[1*_q] = w[1*_q] * nl[1*_q];
      }
      _tmp25 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _k = 0; _k < 6; ++_k) {
          _tmp25[1*_k + 6*_q] = e(1)[1*_k + 6*_q] * _tmp24[1*_q];
        }
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp25));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(e(0)));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(1,0));
        _mapC.noalias() += c21*_mapA*_mapB.transpose();
      }
          
      _tmp29 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp29[1*_i + 2*_q] = w[1*_q] * n[1*_i + 2*_q];
        }
      }
      _tmp30 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_x(1) + 12*_q;
        double const* _B = _tmp29 + 2*_q;
        double * _C = _tmp30 + 6*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp30));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(e(1)));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(1,1));
        _mapC = c01*_mapA*_mapB.transpose();
      }
          
      _tmp33 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp33[1*_i + 2*_q] = w[1*_q] * n[1*_i + 2*_q];
        }
      }
      _tmp34 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_x(1) + 12*_q;
        double const* _B = _tmp33 + 2*_q;
        double * _C = _tmp34 + 6*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(e(1)));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(_tmp34));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(1,1));
        _mapC.noalias() += c11*_mapA*_mapB.transpose();
      }
          
      _tmp37 = _buffer0;
      #pragma omp simd
      for (int _q = 0; _q < 3; ++_q) {
        _tmp37[1*_q] = w[1*_q] * nl[1*_q];
      }
      _tmp38 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _k = 0; _k < 6; ++_k) {
          _tmp38[1*_k + 6*_q] = e(1)[1*_k + 6*_q] * _tmp37[1*_q];
        }
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_tmp38));
        Map<Matrix<double,6,3>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(e(1)));
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapC(a(1,1));
        _mapC.noalias() += c20*_mapA*_mapB.transpose();
      }
          
    }
    constexpr unsigned long const kernel::rhsVolume::NonZeroFlops;
    constexpr unsigned long const kernel::rhsVolume::HardwareFlops;
    void kernel::rhsVolume::execute() {
      assert(E != nullptr);
      assert(F_Q != nullptr);
      assert(J != nullptr);
      assert(W != nullptr);
      assert(b != nullptr);
      double *_tmp0, *_tmp1;
      alignas(16) double _buffer0[7] ;
      alignas(16) double _buffer1[7] ;
      _tmp0 = _buffer0;
      #pragma omp simd
      for (int _q = 0; _q < 7; ++_q) {
        _tmp0[1*_q] = J[1*_q] * W[1*_q];
      }
      _tmp1 = _buffer1;
      #pragma omp simd
      for (int _q = 0; _q < 7; ++_q) {
        _tmp1[1*_q] = F_Q[1*_q] * _tmp0[1*_q];
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,7>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(E));
        Map<Matrix<double,7,1>,Eigen::Unaligned,Stride<7,1>> _mapB(const_cast<double*>(_tmp1));
        Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(b);
        _mapC.noalias() += _mapA*_mapB;
      }
          
    }
    constexpr unsigned long const kernel::rhsFacet::NonZeroFlops;
    constexpr unsigned long const kernel::rhsFacet::HardwareFlops;
    void kernel::rhsFacet::execute() {
      assert(!std::isnan(c10));
      assert(!std::isnan(c20));
      assert(K != nullptr);
      assert(b != nullptr);
      assert(d_xi(0) != nullptr);
      assert(e(0) != nullptr);
      assert(em(0) != nullptr);
      assert(f_q != nullptr);
      assert(g(0) != nullptr);
      assert(n != nullptr);
      assert(nl != nullptr);
      assert(w != nullptr);
      double *_tmp0, *_tmp1, *_tmp2, *_tmp3, *_tmp4, *_tmp7, *_tmp8;
      alignas(16) double _buffer0[6] ;
      alignas(16) double _buffer1[6] ;
      alignas(16) double _buffer2[3] ;
      _tmp0 = _buffer0;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,3,6>,Eigen::Unaligned,Stride<3,1>> _mapA(const_cast<double*>(em(0)));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(K));
        Map<Matrix<double,3,1>,Eigen::Unaligned,Stride<3,1>> _mapC(_tmp0);
        _mapC = _mapA*_mapB;
      }
          
      _tmp1 = _buffer1;
      #pragma omp simd
      for (int _q = 0; _q < 3; ++_q) {
        _tmp1[1*_q] = w[1*_q] * f_q[1*_q];
      }
      _tmp2 = _buffer2;
      #pragma omp simd
      for (int _q = 0; _q < 3; ++_q) {
        _tmp2[1*_q] = _tmp0[1*_q] * _tmp1[1*_q];
      }
      _tmp3 = _buffer0;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _i = 0; _i < 2; ++_i) {
          _tmp3[1*_i + 2*_q] = n[1*_i + 2*_q] * _tmp2[1*_q];
        }
      }
      _tmp4 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = g(0) + 4*_q;
        double const* _B = _tmp3 + 2*_q;
        double * _C = _tmp4 + 2*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,2,2>,Eigen::Aligned16,Stride<2,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,1>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,2,1>,Eigen::Aligned16,Stride<2,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(d_xi(0)));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(_tmp4));
        Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(b);
        _mapC.noalias() += c10*_mapA*_mapB;
      }
          
      _tmp7 = _buffer2;
      #pragma omp simd
      for (int _q = 0; _q < 3; ++_q) {
        _tmp7[1*_q] = w[1*_q] * nl[1*_q];
      }
      _tmp8 = _buffer0;
      #pragma omp simd
      for (int _q = 0; _q < 3; ++_q) {
        _tmp8[1*_q] = f_q[1*_q] * _tmp7[1*_q];
      }
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,3>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(e(0)));
        Map<Matrix<double,3,1>,Eigen::Unaligned,Stride<3,1>> _mapB(const_cast<double*>(_tmp8));
        Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(b);
        _mapC.noalias() += c20*_mapA*_mapB;
      }
          
    }
    constexpr unsigned long const kernel::grad_u::NonZeroFlops;
    constexpr unsigned long const kernel::grad_u::HardwareFlops;
    void kernel::grad_u::execute() {
      assert(d_x(0) != nullptr);
      assert(d_x(1) != nullptr);
      assert(d_xi(0) != nullptr);
      assert(d_xi(1) != nullptr);
      assert(e_q_T != nullptr);
      assert(em(0) != nullptr);
      assert(em(1) != nullptr);
      assert(g(0) != nullptr);
      assert(g(1) != nullptr);
      assert(grad_u != nullptr);
      assert(k(0) != nullptr);
      assert(k(1) != nullptr);
      assert(minv != nullptr);
      assert(u(0) != nullptr);
      assert(u(1) != nullptr);
      assert(w != nullptr);
      double *_tmp0, *_tmp1, *_tmp3, *_tmp4, *_tmp6, *_tmp9, *_tmp10;
      alignas(16) double _buffer0[6] ;
      alignas(16) double _buffer1[12] ;
      _tmp0 = _buffer0;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,3,6>,Eigen::Unaligned,Stride<3,1>> _mapA(const_cast<double*>(em(0)));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(k(0)));
        Map<Matrix<double,3,1>,Eigen::Unaligned,Stride<3,1>> _mapC(_tmp0);
        _mapC = _mapA*_mapB;
      }
          
      _tmp1 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        for (int _i = 0; _i < 2; ++_i) {
          #pragma omp simd
          for (int _e = 0; _e < 2; ++_e) {
            _tmp1[1*_e + 2*_i + 4*_q] = g(0)[1*_e + 2*_i + 4*_q] * _tmp0[1*_q];
          }
        }
      }
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_xi(0) + 12*_q;
        double const* _B = _tmp1 + 4*_q;
        double * _C = d_x(0) + 12*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,2>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      _tmp3 = _buffer0;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,3,6>,Eigen::Unaligned,Stride<3,1>> _mapA(const_cast<double*>(em(1)));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(k(1)));
        Map<Matrix<double,3,1>,Eigen::Unaligned,Stride<3,1>> _mapC(_tmp3);
        _mapC = _mapA*_mapB;
      }
          
      _tmp4 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        for (int _i = 0; _i < 2; ++_i) {
          #pragma omp simd
          for (int _e = 0; _e < 2; ++_e) {
            _tmp4[1*_e + 2*_i + 4*_q] = g(1)[1*_e + 2*_i + 4*_q] * _tmp3[1*_q];
          }
        }
      }
      for (int _q = 0; _q < 3; ++_q) {
        double const* _A = d_xi(1) + 12*_q;
        double const* _B = _tmp4 + 4*_q;
        double * _C = d_x(1) + 12*_q;
        {
          using Eigen::Matrix;
          using Eigen::Map;
          using Eigen::Stride;
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(_A));
          Map<Matrix<double,2,2>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(_B));
          Map<Matrix<double,6,2>,Eigen::Aligned16,Stride<6,1>> _mapC(_C);
          _mapC = _mapA*_mapB;
        }
            
      }
      _tmp6 = _buffer0;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(d_x(0)));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(u(0)));
        Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_tmp6);
        _mapC = _mapA.transpose()*_mapB;
      }
          
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,6,6>,Eigen::Aligned16,Stride<6,1>> _mapA(const_cast<double*>(d_x(1)));
        Map<Matrix<double,6,1>,Eigen::Unaligned,Stride<6,1>> _mapB(const_cast<double*>(u(1)));
        Map<Matrix<double,6,1>,Eigen::Aligned16,Stride<6,1>> _mapC(_tmp6);
        _mapC.noalias() += _mapA.transpose()*_mapB;
      }
          
      _tmp9 = _buffer1;
      for (int _q = 0; _q < 3; ++_q) {
        #pragma omp simd
        for (int _p = 0; _p < 2; ++_p) {
          _tmp9[1*_p + 2*_q] = _tmp6[1*_p + 2*_q] * w[1*_q];
        }
      }
      _tmp10 = _buffer0;
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,2,3>,Eigen::Aligned16,Stride<2,1>> _mapA(const_cast<double*>(_tmp9));
        Map<Matrix<double,3,2>,Eigen::Unaligned,Stride<3,1>> _mapB(const_cast<double*>(e_q_T));
        Map<Matrix<double,2,2>,Eigen::Aligned16,Stride<2,1>> _mapC(_tmp10);
        _mapC = _mapA*_mapB;
      }
          
      {
        using Eigen::Matrix;
        using Eigen::Map;
        using Eigen::Stride;
        Map<Matrix<double,2,2>,Eigen::Aligned16,Stride<2,1>> _mapA(const_cast<double*>(_tmp10));
        Map<Matrix<double,2,2>,Eigen::Unaligned,Stride<2,1>> _mapB(const_cast<double*>(minv));
        Map<Matrix<double,2,2>,Eigen::Aligned16,Stride<2,1>> _mapC(grad_u);
        _mapC = 0.5*_mapA*_mapB;
      }
          
    }
  } // namespace poisson
} // namespace tndm
