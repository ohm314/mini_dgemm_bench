#include <iostream>
#include <chrono>
#include "mkl.h"

#define fix_lda(x)   (((x + 255) & ~255) + 16)
#define min(x,y) (((x) < (y)) ? (x) : (y))

int main(int argc, char **argv) {
  double *A, *B, *C;
  double alpha, beta;
  int m, n, k;
  int niter = 100;

  alpha = 1.0;
  beta = 1.0;
  m = 8000;
  k = fix_lda(4096);
  n = fix_lda(4096);

  double gflop = (2.0*m*k*n)*1e-9*niter;

  A = (double*)mkl_malloc(m*k*sizeof(double), 64);
  B = (double*)mkl_malloc(k*n*sizeof(double), 64);
  C = (double*)mkl_malloc(m*n*sizeof(double), 64);
  if ((A == NULL) || (B == NULL) || (C == NULL)) {
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    return 1;
  }

  for(int i = 0; i < m*k; ++i) {
    A[i] = 1.1*(i+1);
  }
  for(int i = 0; i < k*n; ++i) {
    B[i] = 1.2*(i+2);
  }
  for(int i = 0; i < m*n; ++i) {
    C[i] = 0.0;
  }

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              m, n, k, alpha, A, k, B, n, beta, C, n);

  auto tstart = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < niter; iter++) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, alpha, A, k, B, n, beta, C, n);
  }
  auto tend = std::chrono::high_resolution_clock::now();

  std::chrono::duration<double> tdiff = tend - tstart;
  std::cout << "Time elapsed for " << niter << " iterations: " << tdiff.count() << "s\n";
  std::cout << gflop/tdiff.count() << " GFLOP/s\n";

#ifdef DBG_PRINT
  for(int i = 0; i < min(m, 5); ++i) {
    for(int j = 0; j < min(n, 5); ++j) {
      std::cout << C[j + i*n] << "  ";
    }
    std::cout << std::endl;
  }
#endif
  double chk;
  double sgn = 1.0;
  for(int i = 0; i < m*n; ++i) {
    sgn *= -1.0;
    chk += sgn*C[i];
  }
  std::cout << "Check value: " << chk << std::endl;

  mkl_free(A);
  mkl_free(B);
  mkl_free(C);

  return 0;
}
