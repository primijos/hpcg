
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
 @file ComputeDotProduct.cpp

 HPCG routine
 */

#include "ComputeDotProduct.hpp"
#ifndef HPCG_NO_MPI
#include <mpi.h>
#include "mytimer.hpp"
#endif

/*!
  Routine to compute the dot product of two vectors.

  This routine calls the reference dot-product implementation by default, but
  can be replaced by a custom routine that is optimized and better suited for
  the target system.

  @param[in]  n the number of vector elements (on this processor)
  @param[in]  x, y the input vectors
  @param[out] result a pointer to scalar value, on exit will contain the result.
  @param[out] time_allreduce the time it took to perform the communication between processes
  @param[out] isOptimized should be set to false if this routine uses the reference implementation (is not optimized); otherwise leave it unchanged

  @return returns 0 upon success and non-zero otherwise

  @see ComputeDotProduct_ref
*/

#include "defs.fpga.h"

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([DOTPRODUCT_BLOCK]xv,[DOTPRODUCT_BLOCK]yv) copy_inout([1]result) localmem_copies
#endif
#pragma omp task inout([1]result) in([DOTPRODUCT_BLOCK]xv,[DOTPRODUCT_BLOCK]yv) no_copy_deps
void compute_dot_product_fpga_block(int whoami, double *xv,double *yv, double *result) {
	double local_result = 0.0;
  if (yv==xv) {
    for (local_int_t i=0; i<DOTPRODUCT_BLOCK; i++) local_result += xv[i]*xv[i];
  } else {
    for (local_int_t i=0; i<DOTPRODUCT_BLOCK; i++) local_result += xv[i]*yv[i];
  }
	if (whoami==0)
		*result = local_result;
	else
		*result += local_result;
}

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([n]xv,[n]yv) copy_inout([1]result)
#endif
#pragma omp task inout([1]result) in([n]xv,[n]yv)
void compute_dot_product_fpga(const local_int_t n,double *xv,double *yv, double *result) {
	local_int_t nblocks = n / DOTPRODUCT_BLOCK;
	// XXX TODO check for block sizes non-divisible by n
	int remainder = n % DOTPRODUCT_BLOCK;

	for (local_int_t i=0;i<nblocks;i++) {
		double *_xv = xv + i*DOTPRODUCT_BLOCK;
		double *_yv = yv + i*DOTPRODUCT_BLOCK;
		compute_dot_product_fpga_block(i,_xv,_yv,result);
	}
#pragma omp taskwait
}

int ComputeDotProduct_nw(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double * xv = x.values;
  double * yv = y.values;
	// XXX TODO check for block sizes non-divisible by n
	assert(n % DOTPRODUCT_BLOCK == 0);

	compute_dot_product_fpga(n,xv,yv,&result);
#pragma omp taskwait noflush

#ifndef HPCG_NO_MPI
#pragma omp taskwait on(result)
  double local_result = result;
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  //result = local_result;
#endif

  return 0;
}

int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

		ComputeDotProduct_nw(n,x,y,result,time_allreduce,isOptimized);
#pragma omp taskwait
		return 0;
}
