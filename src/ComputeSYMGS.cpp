
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

#include "ComputeSYMGS.hpp"

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
		copy_in([Alen]AmatrixValues,[Alen]AmtxIndL,[nrow]AnonzerosInRow,[nrow]matrixDiagonalI,[rl]rv) copy_inout([xl]xv)
#endif
#pragma omp task inout([xl]xv) in([Alen]AmatrixValues,[Alen]AmtxIndL,[nrow]AnonzerosInRow,[nrow]matrixDiagonalI,[rl]rv)
void compute_symgs_fpga(local_int_t nrow, local_int_t Alen, double *AmatrixValues, local_int_t *AmtxIndL, char* AnonzerosInRow, int *matrixDiagonalI, const double *rv, local_int_t rl, double *xv, local_int_t xl) {
	local_int_t numberOfNonzerosPerRow = 27;

  for (local_int_t i=0; i< nrow; i++) {
		const double * const currentValues = AmatrixValues + i*numberOfNonzerosPerRow;
		const local_int_t * const currentColIndices = AmtxIndL + i*numberOfNonzerosPerRow;
		const char currentNumberOfNonzeros = AnonzerosInRow[i];
    const double  currentDiagonal = currentValues[matrixDiagonalI[i]]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j=0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j] * xv[curCol];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;

  }

  // Now the back sweep.

  for (local_int_t i=nrow-1; i>=0; i--) {
		const double * const currentValues = AmatrixValues + i*numberOfNonzerosPerRow;
		const local_int_t * const currentColIndices = AmtxIndL + i*numberOfNonzerosPerRow;
		const char currentNumberOfNonzeros = AnonzerosInRow[i];
    const double  currentDiagonal = currentValues[matrixDiagonalI[i]]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j = 0; j< currentNumberOfNonzeros; j++) {
      local_int_t curCol = currentColIndices[j];
      sum -= currentValues[j]*xv[curCol];
    }
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

    xv[i] = sum/currentDiagonal;
  }
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
	local_int_t numberOfNonzerosPerRow = 27;

	compute_symgs_fpga(nrow,A.localNumberOfRows*numberOfNonzerosPerRow,AmatrixValues,AmtxIndL,AnonzerosInRow,matrixDiagonalI,rv,rl,xv,xl);

#pragma omp taskwait noflush
  return 0;
}

int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

	ComputeSYMGS_nw(A,r,x);
#pragma omp taskwait

  return 0;
}
