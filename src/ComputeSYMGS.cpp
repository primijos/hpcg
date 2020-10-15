
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
#include "ComputeSYMGS_ref.hpp"

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
int ComputeSYMGS( const SparseMatrix & A, const Vector & r, Vector & x) {

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;
	double * xv_tmp = ((double **)A.optimizationData)[0];
	int * i_before_diagonal = ((int **)A.optimizationData)[1];
	int * i_after_diagonal = ((int **)A.optimizationData)[2];

	int block_size=1024;
	for (local_int_t ii=0; ii<nrow; ii+=block_size) {
		local_int_t nnrow = ii+block_size;
		nnrow = nnrow < nrow ? nnrow : nrow;
#ifndef HPCG_NO_OPENMP
	#pragma omp parallel for
#endif
		for (local_int_t i=ii; i< nnrow; i++) {
			const double * const currentValues = A.matrixValues[i];
			const local_int_t * const currentColIndices = A.mtxIndL[i];
			const int nnz_to_the_left = i_before_diagonal[i];
			const int currentNumberOfNonzeros = A.nonzerosInRow[i];
	
			double sum = rv[i]; // RHS value
	
			for (int j=nnz_to_the_left+1; j < currentNumberOfNonzeros; j++) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
	
			xv_tmp[i] = sum;
	
		}
	
		for (local_int_t i=ii; i< nnrow; i++) {
			const double * const currentValues = A.matrixValues[i];
			const local_int_t * const currentColIndices = A.mtxIndL[i];
			const int nnz_to_the_left = i_before_diagonal[i];
			const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
			double sum = xv_tmp[i]; // RHS value
	
			for (int j=0; j< nnz_to_the_left; j++) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j] * xv[curCol];
			}
	
			xv[i] = sum/currentDiagonal;
	
		}
	}

  // Now the back sweep.
	//
	for (local_int_t ii=nrow-1; ii>=0; ii-=block_size) {
		local_int_t low_lim = ii-block_size;
		low_lim = low_lim < 0 ? 0 : low_lim;
#ifndef HPCG_NO_OPENMP
	#pragma omp parallel for
#endif
		for (local_int_t i=ii; i>=low_lim; i--) {
			const double * const currentValues = A.matrixValues[i];
			const local_int_t * const currentColIndices = A.mtxIndL[i];
			const int nnz_to_the_left = i_before_diagonal[i];
			const int currentNumberOfNonzeros = A.nonzerosInRow[i];
			double sum = rv[i]; // RHS value
	
			for (int j = 0; j< nnz_to_the_left; j++) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j]*xv[curCol];
			}
	
			xv_tmp[i] = sum;
		}
	
		for (local_int_t i=ii; i>=low_lim; i--) {
			const double * const currentValues = A.matrixValues[i];
			const local_int_t * const currentColIndices = A.mtxIndL[i];
			const int nnz_to_the_left = i_before_diagonal[i];
			const int currentNumberOfNonzeros = A.nonzerosInRow[i];
			const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
			double sum = xv_tmp[i]; // RHS value
	
			for (int j = nnz_to_the_left + 1; j< currentNumberOfNonzeros; j++) {
				local_int_t curCol = currentColIndices[j];
				sum -= currentValues[j]*xv[curCol];
			}
	
			xv[i] = sum/currentDiagonal;
		}
	}

  return 0;
}
