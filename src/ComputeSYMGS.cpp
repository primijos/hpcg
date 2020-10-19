
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

#include <stdio.h>
#include "ComputeSYMGS.hpp"
#include "ComputeSYMGS_ref.hpp"

#pragma omp task
void do_forward(void *const &optimizationData,const double * const &rv, double * const &xv, double **const &matrixValues, local_int_t **const &mtxIndL, char *const &nonzerosInRow, double **const &matrixDiagonal, int &block_size, int &child_block_size, int &nblocks, int cur_block);

#pragma omp task
void do_backward(void *const &optimizationData,const double * const &rv, double * const &xv, double **const &matrixValues, local_int_t **const &mtxIndL, char *const &nonzerosInRow, double **const &matrixDiagonal, int &block_size, int &child_block_size, int &nblocks, int cur_block);

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


  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

#ifndef HPCG_NO_MPI
  ExchangeHalo(A,x);
#endif

  const local_int_t nrow = A.localNumberOfRows;
  double ** matrixDiagonal = A.matrixDiagonal;  // An array of pointers to the diagonal entries A.matrixValues
  const double * const rv = r.values;
  double * const xv = x.values;
	//printf("xxx xv is %p .. %p\n",xv,&xv[nrow-1]);

	int block_size=1024;
	int first_block_size=block_size;
	int nblocks=nrow/block_size;
	if (nblocks==0) {
		nblocks=1;
		first_block_size=block_size=nrow;
	}
	if (nblocks*block_size!=nrow) {
		first_block_size = block_size + (nrow-nblocks*block_size);
	}
	//printf("Matrix %p nblocks=%d block_size=%d first_block_size=%d nrow=%d\n",&A,nblocks,block_size,first_block_size,nrow);
  assert((nblocks-1)*block_size+first_block_size==nrow); // Make sure nrow is multiple of block_size
// lauch first block, will launch the rest
	//printf("ComputeSYMGS nrow=%d block_size=%d nblocks=%d\n",nrow,block_size,nblocks);
	//printf("Launching first\n");
	do_forward(A.optimizationData,rv,xv,A.matrixValues,A.mtxIndL,A.nonzerosInRow,A.matrixDiagonal,first_block_size,block_size,nblocks,nblocks-1);
#pragma omp taskwait
  // Now the back sweep.
	//printf("Launching backward!\n");
	do_backward(A.optimizationData,rv,xv,A.matrixValues,A.mtxIndL,A.nonzerosInRow,A.matrixDiagonal,first_block_size,block_size,nblocks,0);
#pragma omp taskwait

	/*
	//printf("Doing the back sweep %d\n",nrow);
  for (local_int_t i=nrow-1; i>=0; i--) {
    const double * const currentValues = A.matrixValues[i];
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value

    for (int j = 0; j< currentNumberOfNonzeros; j++) {
			//printf("back sweep i=%d j=%d\n",i,j);
      local_int_t curCol = currentColIndices[j];
			//printf("A\n");
      sum -= currentValues[j]*xv[curCol];
    }
			//printf("B\n");
    sum += xv[i]*currentDiagonal; // Remove diagonal contribution from previous loop

			//printf("C\n");
    xv[i] = sum/currentDiagonal;
			//printf("D\n");
  }
	*/

  return 0;
}


void do_forward(void *const &optimizationData,const double * const &rv, double * const &xv, double **const &matrixValues, local_int_t **const &mtxIndL, char *const &nonzerosInRow, double **const &matrixDiagonal, int &block_size, int &child_block_size, int &nblocks, int cur_block) {
	double xv_tmp[block_size];

	//printf("Going forward %d\n",cur_block);
	if (cur_block != 0) {
		//printf("Going forward %d launching child\n",cur_block);
		do_forward(optimizationData,rv,xv,matrixValues,mtxIndL,nonzerosInRow,matrixDiagonal,child_block_size,child_block_size,nblocks,cur_block-1);
	}

	local_int_t block_start=child_block_size*cur_block;
	local_int_t block_end=block_start+block_size;

	//printf("Going forward %d first half\n",cur_block);
	//printf("Going forward %d first half start=%d end=%d\n",cur_block,block_start,block_end);
  for (local_int_t i=block_start,c=0; i< block_end; i++,c++) {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value
		int j_after_diagonal = ((int **)optimizationData)[1][i];

		//printf("Going forward %d first half, cnnz=%d\n",cur_block,currentNumberOfNonzeros);
    for (int j=j_after_diagonal;j< currentNumberOfNonzeros; j++) {
			////printf("Going forward %d first half, i=%d j=%d\n",cur_block,i,j);
      local_int_t curCol = currentColIndices[j];
			//printf("Updating i=%05d j=%05d nnz=%d [first half]\n",i,j,currentNumberOfNonzeros);
      sum -= currentValues[j] * xv[curCol];
    }
		//printf("OK block_size=%d i=%d c=%d\n",block_size,i,c);
		xv_tmp[c] = sum;
		//printf("OKK\n");
	}
#pragma omp taskwait

	//printf("Going forward %d second half\n",cur_block);
  for (local_int_t i=block_start,c=0; i< block_end; i++,c++) {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = xv_tmp[c]; // RHS value
		int j_before_diagonal = ((int **)optimizationData)[0][i];

    for (int j=0; j<=j_before_diagonal; j++) {
      local_int_t curCol = currentColIndices[j];
			//printf("Updating i=%05d j=%05d [second half]\n",i,j);
      sum -= currentValues[j] * xv[curCol];
    }
		//printf("xxx Updating xv[i]=%p\n",&xv[i]);
		xv[i] = sum/currentDiagonal;
	}
	//printf("Forward done %d\n",cur_block);
}

void do_backward(void *const &optimizationData,const double * const &rv, double * const &xv, double **const &matrixValues, local_int_t **const &mtxIndL, char *const &nonzerosInRow, double **const &matrixDiagonal, int &block_size, int &child_block_size, int &nblocks, int cur_block) {
	double xv_tmp[block_size];

	int my_block_size = child_block_size;
	//printf("Going backward %d\n",cur_block);
	if (cur_block != nblocks-1) {
		//printf("Going backward %d launching child\n",cur_block);
		do_backward(optimizationData,rv,xv,matrixValues,mtxIndL,nonzerosInRow,matrixDiagonal,block_size,child_block_size,nblocks,cur_block+1);
	} else {
		my_block_size = block_size;
	}

	local_int_t block_start=child_block_size*cur_block;
	local_int_t block_end=block_start+my_block_size;

	//printf("Going backward %d first half\n",cur_block);
	//printf("Going backward %d first half start=%d end=%d\n",cur_block,block_end,block_start);
  for (local_int_t i=block_end-1,c=my_block_size-1; i>= block_start; i--,c--) {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = rv[i]; // RHS value
		int j_before_diagonal = ((int **)optimizationData)[0][i];

    for (int j=0; j<=j_before_diagonal; j++) {
      local_int_t curCol = currentColIndices[j];
			//printf("Updating i=%05d j=%05d [first half]\n",i,j);
      sum -= currentValues[j] * xv[curCol];
    }
		////printf("xxx Updating xv[i]=%p\n",&xv[i]);
		xv_tmp[c] = sum;
	}
#pragma omp taskwait

	//printf("Going backward %d second half\n",cur_block);
	//printf("Going backward %d second half start=%d end=%d\n",cur_block,block_end,block_start);
  for (local_int_t i=block_end-1,c=my_block_size-1; i>= block_start; i--,c--) {
    const double * const currentValues = matrixValues[i];
    const local_int_t * const currentColIndices = mtxIndL[i];
    const int currentNumberOfNonzeros = nonzerosInRow[i];
    const double  currentDiagonal = matrixDiagonal[i][0]; // Current diagonal value
    double sum = xv_tmp[c]; // RHS value
		int j_after_diagonal = ((int **)optimizationData)[1][i];

		////printf("Going backward %d second half, cnnz=%d\n",cur_block,currentNumberOfNonzeros);
    for (int j=j_after_diagonal;j< currentNumberOfNonzeros; j++) {
			////printf("Going backward %d second half, i=%d j=%d\n",cur_block,i,j);
      local_int_t curCol = currentColIndices[j];
			//printf("Updating i=%05d j=%05d nnz=%d [second half]\n",i,j,currentNumberOfNonzeros);
      sum -= currentValues[j] * xv[curCol];
    }
		//printf("OK block_size=%d i=%d c=%d\n",my_block_size,i,c);
		//printf("OKK\n");
		xv[i] = sum/currentDiagonal;
	}
	//printf("Backward done %d\n",cur_block);
}
