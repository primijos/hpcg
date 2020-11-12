
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
 @file ComputeSPMV.cpp

 HPCG routine
 */

#include "ComputeSPMV.hpp"
#include "ComputeSPMV_ref.hpp"

/*!
  Routine to compute sparse matrix vector product y = Ax where:
  Precondition: First call exchange_externals to get off-processor values of x

  This routine calls the reference SpMV implementation by default, but
  can be replaced by a custom, optimized routine suited for
  the target system.

  @param[in]  A the known system matrix
  @param[in]  x the known vector
  @param[out] y the On exit contains the result: Ax.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeSPMV_ref
*/
#include "defs.fpga.h"
#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([Alen]AmatrixValues,[Alen]AmtxIndL,[nrow]AnonzerosInRow,[xl]xv) copy_inout([yl]yv)
#endif
#pragma omp task inout([yl]yv) in([Alen]AmatrixValues,[Alen]AmtxIndL,[nrow]AnonzerosInRow,[xl]xv)
void compute_spmv_fpga(local_int_t nrow, local_int_t Alen, double *AmatrixValues, local_int_t *AmtxIndL,char *AnonzerosInRow, double *yv, local_int_t yl, const double * const xv, local_int_t xl) {
	local_int_t numberOfNonzerosPerRow = 27;

  for (local_int_t i=0; i< nrow; i++)  {
    double sum = 0.0;
    const double * const cur_vals = AmatrixValues + i*numberOfNonzerosPerRow;
    const local_int_t * const cur_inds = AmtxIndL + i*numberOfNonzerosPerRow;
    const char cur_nnz = AnonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
}
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {

  assert(x.localLength>=A.localNumberOfColumns); // Test vector lengths
  assert(y.localLength>=A.localNumberOfRows);

#ifndef HPCG_NO_MPI
    ExchangeHalo(A,x);
#endif
  const double * const xv = x.values;
  double * const yv = y.values;
  const local_int_t nrow = A.localNumberOfRows;

	local_int_t xl = x.localLength;
	local_int_t yl = y.localLength;
	char * AnonzerosInRow = A.nonzerosInRow;
	double *AmatrixValues = A.matrixValues[0];
	local_int_t *AmtxIndL = A.mtxIndL[0];
	local_int_t numberOfNonzerosPerRow = 27;

	compute_spmv_fpga(nrow,A.localNumberOfRows*numberOfNonzerosPerRow,AmatrixValues,AmtxIndL,AnonzerosInRow,yv,yl,xv,xl);
#pragma omp taskwait
  return 0;
}
