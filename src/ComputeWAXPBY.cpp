
//@HEADER
// ***************************************************
//
// HPCG: High Performance Conjugate Gradient Benchmark
//
// Contact:
// Michael A. Heroux ( maherou@sandia.gov)
// Jack Dongarra     (dongarra@eecs.utk.edu)
// Piotr Luszczek    (luszczek@eecs.utk.edu)
//
// ***************************************************
//@HEADER

/*!
 @file ComputeWAXPBY.cpp

 HPCG routine
 */

#include "ComputeWAXPBY.hpp"

/*!
  Routine to compute the update of a vector with the sum of two
  scaled vectors where: w = alpha*x + beta*y

  This routine calls the reference WAXPBY implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in] n the number of vector elements (on this processor)
  @param[in] alpha, beta the scalars applied to x and y respectively.
  @param[in] x, y the input vectors
  @param[out] w the output vector
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeWAXPBY_ref
*/

#include "defs.fpga.h"
#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([WAXPBY_BLOCK]xv,[WAXPBY_BLOCK]yv) copy_inout([WAXPBY_BLOCK]wv) localmem_copies
#endif
#pragma omp task inout([WAXPBY_BLOCK]wv) in([WAXPBY_BLOCK]xv,[WAXPBY_BLOCK]yv) no_copy_deps
void compute_waxpby_fpga_block(double alpha, double *xv, double beta, double *yv, double *wv) {
  if (alpha==1.0) {
    for (local_int_t i=0; i<WAXPBY_BLOCK; i++) wv[i] = xv[i] + beta * yv[i];
  } else if (beta==1.0) {
    for (local_int_t i=0; i<WAXPBY_BLOCK; i++) wv[i] = alpha * xv[i] + yv[i];
  } else  {
    for (local_int_t i=0; i<WAXPBY_BLOCK; i++) wv[i] = alpha * xv[i] + beta * yv[i];
  }
}

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	copy_in([n_x]xv,[n_y]yv) copy_inout([n_w]wv)
#endif
#pragma omp task in([n_x]xv,[n_y]yv) inout([n_w]wv)
void compute_waxpby_fpga(local_int_t n, local_int_t n_x, local_int_t n_y, local_int_t n_w, double alpha, double *xv, double beta, double *yv, double *wv) {
	local_int_t nblocks = n / WAXPBY_BLOCK;
	// XXX TODO check for block sizes non-divisible by n
	int remainder = n % WAXPBY_BLOCK;

	for (local_int_t i=0;i<nblocks;i++) {
		double *_xv = xv + i*WAXPBY_BLOCK;
		double *_yv = yv + i*WAXPBY_BLOCK;
		double *_wv = wv + i*WAXPBY_BLOCK;
		compute_waxpby_fpga_block(alpha,_xv,beta,_yv,_wv);
	}
#pragma omp taskwait
}

int ComputeWAXPBY_nw(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double * xv = x.values;
  double * yv = y.values;
  double * wv = w.values;
	// XXX TODO check for block sizes non-divisible by n
	assert(n % WAXPBY_BLOCK==0);

	compute_waxpby_fpga(n,x.localLength,y.localLength,w.localLength,alpha,xv,beta,yv,wv);
#pragma omp taskwait noflush

  return 0;
}

int ComputeWAXPBY(const local_int_t n, const double alpha, const Vector & x,
    const double beta, const Vector & y, Vector & w, bool & isOptimized) {

		ComputeWAXPBY_nw(n,alpha,x,beta,y,w,isOptimized);
#pragma omp taskwait

  return 0;
}
