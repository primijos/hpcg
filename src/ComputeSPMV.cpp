
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
#include <stdio.h>

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
	 copy_in([SPMV_BLOCK_M]AmatrixValues,[SPMV_BLOCK_M]AmtxIndL,[SPMV_BLOCK]AnonzerosInRow) copy_inout([SPMV_BLOCK]yv) localmem_copies
#endif
#pragma omp task inout([SPMV_BLOCK]yv) in([SPMV_BLOCK_M]AmatrixValues,[SPMV_BLOCK_M]AmtxIndL,[SPMV_BLOCK]AnonzerosInRow,[xl]xv) no_copy_deps
void compute_spmv_fpga_block(double *AmatrixValues, local_int_t *AmtxIndL, char *AnonzerosInRow, double *yv, const double * const xv, local_int_t xl) {

  for (local_int_t i=0; i<SPMV_BLOCK; i++)  {
    double sum = 0.0;
    const double * const cur_vals = AmatrixValues + i*NNZ_PER_ROW;
    const local_int_t * const cur_inds = AmtxIndL + i*NNZ_PER_ROW;
    const char cur_nnz = AnonzerosInRow[i];

    for (int j=0; j< cur_nnz; j++)
      sum += cur_vals[j]*xv[cur_inds[j]];
    yv[i] = sum;
  }
}

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([Alen]AmatrixValues,[Alen]AmtxIndL,[nrow]AnonzerosInRow,[xl]xv) copy_inout([yl]yv)
#endif
#pragma omp task inout([yl]yv) in([Alen]AmatrixValues,[Alen]AmtxIndL,[nrow]AnonzerosInRow,[xl]xv)
void compute_spmv_fpga(local_int_t nrow, local_int_t Alen, double *AmatrixValues, local_int_t *AmtxIndL,char *AnonzerosInRow, double *yv, local_int_t yl, const double * const xv, local_int_t xl) {

	local_int_t nblocks = nrow / SPMV_BLOCK;
	// XXX TODO check for block sizes non-divisible by n*nnz
	// XXX TODO check for block sizes non-divisible by n
	int remainder = nrow % SPMV_BLOCK;

	for (local_int_t i=0;i<nblocks;i++) {
		double *_AmatrixValues = AmatrixValues + i*SPMV_BLOCK_M;
		local_int_t *_AmtxIndL = AmtxIndL + i*SPMV_BLOCK_M;
		char *_AnonzerosInRow = AnonzerosInRow + i*SPMV_BLOCK;
		double *_yv = yv + i*SPMV_BLOCK;

		compute_spmv_fpga_block(_AmatrixValues,_AmtxIndL,_AnonzerosInRow,_yv,xv,xl);
	}
}

int ComputeSPMV_nw( const SparseMatrix & A, Vector & x, Vector & y) {
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
	// XXX TODO check for block sizes non-divisible by n
	assert(nrow % SPMV_BLOCK == 0);

	compute_spmv_fpga(nrow,A.localNumberOfRows*NNZ_PER_ROW,AmatrixValues,AmtxIndL,AnonzerosInRow,yv,yl,xv,xl);
#pragma omp taskwait noflush
  return 0;
}
int ComputeSPMV( const SparseMatrix & A, Vector & x, Vector & y) {
	ComputeSPMV_nw(A,x,y);
	#pragma omp taskwait
	return 0;
}
