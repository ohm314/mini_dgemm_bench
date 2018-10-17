#include <iostream>
#include <chrono>
#include "mkl.h"
#include "omp.h"

#define fix_lda(x)   (((x + 255) & ~255) + 16)
#define min(x,y) (((x) < (y)) ? (x) : (y))

#define ALPHA 1.0
#define BETA 1.0
#define NITER 100
#define DIM_M 8000
#define DIM_K 4096
#define DIM_N 4096

int main(int argc, char **argv) {
  double alpha, beta;
  size_t m, n, k;
  int niter = NITER;
  int nthreads = omp_get_max_threads();
#ifdef DBG_PRINT
  std::cout << "Running with " << nthreads << " outer threads\n";
#endif
  alpha = ALPHA;
  beta = BETA;
  m = DIM_M;
  k = fix_lda(DIM_K);
  n = fix_lda(DIM_N);

  omp_set_max_active_levels(2);

  double gflop = (m*n*(2.0*k+2.0))*1e-9*niter*nthreads;
  auto As = (double**)malloc(nthreads*sizeof(double*));
  auto Bs = (double**)malloc(nthreads*sizeof(double*));
  auto Cs = (double**)malloc(nthreads*sizeof(double*));

  for (int ithrd = 0; ithrd < nthreads; ithrd++) {
    auto A = (double*)mkl_malloc(m*k*sizeof(double), 64);
    auto B = (double*)mkl_malloc(k*n*sizeof(double), 64);
    auto C = (double*)mkl_malloc(m*n*sizeof(double), 64);
    if ((A == NULL) || (B == NULL) || (C == NULL)) {
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }
    As[ithrd] = A;
    Bs[ithrd] = B;
    Cs[ithrd] = C;
  }

  #pragma omp parallel for schedule(static,1) num_threads(nthreads)
  for (int ithrd = 0; ithrd < nthreads; ithrd++) {
    auto A = As[ithrd];
    auto B = Bs[ithrd];
    auto C = Cs[ithrd];
    for(size_t i = 0; i < m*k; ++i) {
      A[i] = 1.1*(i+1);
    }
    for(size_t i = 0; i < k*n; ++i) {
      B[i] = 1.2*(i+2);
    }
    for(size_t i = 0; i < m*n; ++i) {
      C[i] = 0.0;
    }
  }

  #pragma omp parallel for schedule(static, 1) num_threads(nthreads)
  for (int ithrd = 0; ithrd < nthreads; ithrd++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, As[ithrd], k, Bs[ithrd], n,
                beta, Cs[ithrd], n);
  }

  auto tstart = std::chrono::high_resolution_clock::now();
  #pragma omp parallel for schedule(static, 1) num_threads(nthreads)
  for (int ithrd = 0; ithrd < nthreads; ithrd++) {
    for (int iter = 0; iter < niter; iter++) {
      cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  m, n, k, alpha, As[ithrd], k, Bs[ithrd], n,
                  beta, Cs[ithrd], n);
    }
  }
  auto tend = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> tdiff = tend - tstart;
  std::cout << "Time elapsed for " << niter << " iterations: " << tdiff.count() << "s\n";
  std::cout << gflop/tdiff.count() << " GFLOP/s\n";

#ifdef DBG_PRINT
  for(int i = 0; i < min(m, 5); ++i) {
    for(int j = 0; j < min(n, 5); ++j) {
      std::cout << Cs[0][j + i*n] << "  ";
    }
    std::cout << std::endl;
  }
#endif

  double chk;
  double sgn = 1.0;
  for(int ithrd = 0; ithrd < nthreads; ithrd++) {
    for(int i = 0; i < m*n; ++i) {
      sgn *= -1.0;
      chk += sgn*Cs[ithrd][i];
    }
  }
  std::cout << "Check value: " << chk << std::endl;
  for(int ithrd = 0; ithrd < nthreads; ithrd++) {
    mkl_free(As[ithrd]);
    mkl_free(Bs[ithrd]);
    mkl_free(Cs[ithrd]);
  }
  free(As);
  free(Bs);
  free(Cs);

  return 0;
}
