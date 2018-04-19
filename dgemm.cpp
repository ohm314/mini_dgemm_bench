#include <iostream>
#include <chrono>
#include "mkl.h"

#define fix_lda(x)   (((x + 255) & ~255) + 16)
#define min(x,y) (((x) < (y)) ? (x) : (y))

int main(int argc, char **argv) {
  int m, n, k;

  int narrays = 20;
  int niter = 10;
  int ngroups = 2;
  //m = 8000;
  //k = fix_lda(4096);
  //n = fix_lda(4096);
  m = 128;
  k = fix_lda(128);
  n = fix_lda(128);
  double gflop = (2.0*m*k*n)*1e-9*narrays*niter;

  auto groups = (int*)malloc(ngroups*sizeof(int));
  auto ms = (int*)malloc(ngroups*sizeof(int));
  auto ns = (int*)malloc(ngroups*sizeof(int));
  auto ks = (int*)malloc(ngroups*sizeof(int));
  auto alpha = (double*)malloc(ngroups*sizeof(double));
  auto beta = (double*)malloc(ngroups*sizeof(double));
  auto trans = (CBLAS_TRANSPOSE*)malloc(ngroups*sizeof(CBLAS_TRANSPOSE));
  for (int igrp = 0; igrp < ngroups; igrp++) {
    groups[igrp] = narrays / ngroups;
    alpha[igrp] = 1.0;
    beta[igrp] = 1.0;
    ms[igrp] = m;
    ns[igrp] = n;
    ks[igrp] = k;
    trans[igrp] = CblasNoTrans;
  }

  auto **As = (const double**)malloc(narrays*sizeof(double*));
  auto **Bs = (const double**)malloc(narrays*sizeof(double*));
  auto **Cs = (double**)malloc(narrays*sizeof(double*));
  for (int iarr = 0; iarr < narrays; iarr++) {
    double *A = (double*)mkl_malloc(m*k*sizeof(double), 64);
    double *B = (double*)mkl_malloc(k*n*sizeof(double), 64);
    double *C = (double*)mkl_malloc(m*n*sizeof(double), 64);
    if ((A == NULL) || (B == NULL) || (C == NULL)) {
      mkl_free(A);
      mkl_free(B);
      mkl_free(C);
      return 1;
    }
    for(int i = 0; i < m*k; ++i) {
      A[i] = 1.1*(i+1);
      B[i] = 1.2*(i+2);
      C[i] = 0.0;
    }
    As[iarr] = A;
    Bs[iarr] = B;
    Cs[iarr] = C;
  }

  cblas_dgemm_batch(CblasRowMajor, trans, trans,
              ms, ns, ks, alpha, As, ks, Bs, ns, beta, Cs, ns,
              ngroups, groups);

  auto tstart = std::chrono::high_resolution_clock::now();
  for (int iter = 0; iter < niter; iter++) {
    cblas_dgemm_batch(CblasRowMajor, trans, trans,
                ms, ns, ks, alpha, As, ks, Bs, ns, beta, Cs, ns,
                ngroups, groups);
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
  double chk;
  double sgn = 1.0;
  for(int i = 0; i < m*n; ++i) {
    sgn *= -1.0;
    chk += sgn*C[i];
  }
  std::cout << "Check value: " << chk << std::endl;
#endif

  for (int iarr = 0; iarr < narrays; iarr++) {
    mkl_free((double*)As[iarr]);
    mkl_free((double*)Bs[iarr]);
    mkl_free((double*)Cs[iarr]);
  }
  free(As);
  free(Bs);
  free(Cs);
  free(ms);
  free(ns);
  free(ks);

  return 0;
}
