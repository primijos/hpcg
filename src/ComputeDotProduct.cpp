
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
#include "ComputeDotProduct_ref.hpp"

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
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([n]xv,[n]yv) copy_inout([1]result)
#pragma omp task inout([1]result) in([n]xv,[n]yv)
void compute_dot_product_fpga(const local_int_t n,double *xv,double *yv, double *result) {
	double local_result = 0.0;
  if (yv==xv) {
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*xv[i];
  } else {
    for (local_int_t i=0; i<n; i++) local_result += xv[i]*yv[i];
  }
	*result = local_result;
}
int ComputeDotProduct(const local_int_t n, const Vector & x, const Vector & y,
    double & result, double & time_allreduce, bool & isOptimized) {

  assert(x.localLength>=n); // Test vector lengths
  assert(y.localLength>=n);

  double local_result = 0.0;
  double * xv = x.values;
  double * yv = y.values;

	compute_dot_product_fpga(n,xv,yv,&local_result);
#pragma omp taskwait

#ifndef HPCG_NO_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM,
      MPI_COMM_WORLD);
  result = global_result;
  time_allreduce += mytimer() - t0;
#else
  time_allreduce += 0.0;
  result = local_result;
#endif

  return 0;
}
