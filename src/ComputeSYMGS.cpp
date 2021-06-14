
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
 @file ComputeSYMGS.cpp

 HPCG routine
 */

#ifndef HPCG_NO_MPI
#include "ExchangeHalo.hpp"
#endif
#include "ComputeSYMGS.hpp"
#include <stdio.h>

/*!
  Routine to compute one step of symmetric Gauss-Seidel:

  Assumption about the structure of matrix A:
  - Each row 'i' of the matrix has nonzero diagonal value whose address is matrixDiagonal[i]
  - Entries in row 'i' are ordered such that:
       - lower triangular terms are stored before the diagonal element.
       - upper triangular terms are stored after the diagonal element.
       - No other assumptions are made about entry ordering.

  Symmetric Gauss-Seidel notes:
  - We use the input vector x as the RHS and start with an initial guess for y of all zeros.
  - We perform one forward sweep.  Since y is initially zero we can ignore the upper triangular terms of A.
  - We then perform one back sweep.
       - For simplicity we include the diagonal contribution in the for-j loop, then correct the sum after

  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On entry, x should contain relevant values, on exit x contains the result of one symmetric GS sweep with r as the RHS.

  @return returns 0 upon success and non-zero otherwise

  @warning Early versions of this kernel (Version 1.1 and earlier) had the r and x arguments in reverse order, and out of sync with other kernels.

  @see ComputeSYMGS_ref
*/

#include "defs.fpga.h"

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
		copy_in([SYMGS_BLOCK_M]AmatrixValues,[SYMGS_BLOCK_M]AmtxIndL,[SYMGS_BLOCK]AnonzerosInRow,[SYMGS_BLOCK]matrixDiagonalI,[SYMGS_BLOCK]rv) localmem_copies
#endif
#pragma omp task inout([xl]xv) in([SYMGS_BLOCK_M]AmatrixValues,[SYMGS_BLOCK_M]AmtxIndL,[SYMGS_BLOCK]AnonzerosInRow,[SYMGS_BLOCK]matrixDiagonalI,[SYMGS_BLOCK]rv) no_copy_deps
void compute_symgs_fpga_block_fwd(local_int_t block_num, double *AmatrixValues, local_int_t *AmtxIndL, char* AnonzerosInRow, int *matrixDiagonalI, const double *rv, double *xv, local_int_t xl) {
  for (local_int_t i=0; i< SYMGS_BLOCK; i++) {
		local_int_t ii = block_num*SYMGS_BLOCK + i;
		const double * const currentValues = AmatrixValues + i*NNZ_PER_ROW;
		const local_int_t * const currentColIndices = AmtxIndL + i*NNZ_PER_ROW;
		const char currentNumberOfNonzeros = AnonzerosInRow[i];
    const double  currentDiagonal = currentValues[matrixDiagonalI[i]]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j=0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[ii]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[ii] = sum/currentDiagonal;

  }
}

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
		copy_in([SYMGS_BLOCK_M]AmatrixValues,[SYMGS_BLOCK_M]AmtxIndL,[SYMGS_BLOCK]AnonzerosInRow,[SYMGS_BLOCK]matrixDiagonalI,[SYMGS_BLOCK]rv) localmem_copies
#endif
#pragma omp task inout([xl]xv) in([SYMGS_BLOCK_M]AmatrixValues,[SYMGS_BLOCK_M]AmtxIndL,[SYMGS_BLOCK]AnonzerosInRow,[SYMGS_BLOCK]matrixDiagonalI,[SYMGS_BLOCK]rv) no_copy_deps
void compute_symgs_fpga_block_bwd(local_int_t block_num, double *AmatrixValues, local_int_t *AmtxIndL, char* AnonzerosInRow, int *matrixDiagonalI, const double *rv, double *xv, local_int_t xl) {
  for (local_int_t i=SYMGS_BLOCK-1; i>=0; i--) {
		local_int_t ii = block_num*SYMGS_BLOCK + i;
		const double * const currentValues = AmatrixValues + i*NNZ_PER_ROW;
		const local_int_t * const currentColIndices = AmtxIndL + i*NNZ_PER_ROW;
		const char currentNumberOfNonzeros = AnonzerosInRow[i];
    const double  currentDiagonal = currentValues[matrixDiagonalI[i]]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j = 0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j]*xv[curCol];
    }
    sum += xv[ii]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[ii] = sum/currentDiagonal;
  }
}

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
		copy_in([Alen]AmatrixValues,[Alen]AmtxIndL,[nrow]AnonzerosInRow,[nrow]matrixDiagonalI,[rl]rv) copy_inout([xl]xv)
#endif
#pragma omp task inout([xl]xv) in([Alen]AmatrixValues,[Alen]AmtxIndL,[nrow]AnonzerosInRow,[nrow]matrixDiagonalI,[rl]rv)
void compute_symgs_fpga(local_int_t nrow, local_int_t Alen, double *AmatrixValues, local_int_t *AmtxIndL, char* AnonzerosInRow, int *matrixDiagonalI, const double *rv, local_int_t rl, double *xv, local_int_t xl) {
	local_int_t nblocks = nrow / SYMGS_BLOCK;
	// XXX TODO check for block sizes non-divisible by n
	int remainder = nrow % SYMGS_BLOCK;

	for (local_int_t i=0;i<nblocks;i++) {
		double *_AmatrixValues = AmatrixValues + i*SYMGS_BLOCK_M;
		local_int_t *_AmtxIndL = AmtxIndL + i*SYMGS_BLOCK_M;
		char *_AnonzerosInRow = AnonzerosInRow + i*SYMGS_BLOCK;
		int *_matrixDiagonalI = matrixDiagonalI + i*SYMGS_BLOCK;
		const double *_rv = rv + i*SYMGS_BLOCK;

		compute_symgs_fpga_block_fwd(i,_AmatrixValues,_AmtxIndL,_AnonzerosInRow,_matrixDiagonalI,_rv,xv,xl);
	}

  // Now the back sweep.
	for (local_int_t i=nblocks-1;i>=0;i--) {
		double *_AmatrixValues = AmatrixValues + i*SYMGS_BLOCK_M;
		local_int_t *_AmtxIndL = AmtxIndL + i*SYMGS_BLOCK_M;
		char *_AnonzerosInRow = AnonzerosInRow + i*SYMGS_BLOCK;
		int *_matrixDiagonalI = matrixDiagonalI + i*SYMGS_BLOCK;
		const double *_rv = rv + i*SYMGS_BLOCK;

		compute_symgs_fpga_block_bwd(i,_AmatrixValues,_AmtxIndL,_AnonzerosInRow,_matrixDiagonalI,_rv,xv,xl);
	}
#pragma omp taskwait
}

int ComputeSYMGS_nw( const SparseMatrix & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  const double * const rv = r.values;
  double * const xv = x.values;

	local_int_t xl = x.localLength;
	local_int_t rl = r.localLength;
	char * AnonzerosInRow = A.nonzerosInRow;
	double *AmatrixValues = A.matrixValues[0];
	local_int_t *AmtxIndL = A.mtxIndL[0];
	int *matrixDiagonalI = (int*)A.optimizationData;

	compute_symgs_fpga(nrow,A.localNumberOfRows*NNZ_PER_ROW,AmatrixValues,AmtxIndL,AnonzerosInRow,matrixDiagonalI,rv,rl,xv,xl);
#pragma omp taskwait noflush
  return 0;
}

int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

	ComputeSYMGS_nw(A,r,x);
#pragma omp taskwait

  return 0;
}
