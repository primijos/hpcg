
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
 @file OptimizeProblem.cpp

 HPCG routine
 */

#include <stdio.h>
#include "OptimizeProblem.hpp"
/*!
  Optimizes the data structures used for CG iteration to increase the
  performance of the benchmark version of the preconditioned CG algorithm.

  @param[inout] A      The known system matrix, also contains the MG hierarchy in attributes Ac and mgData.
  @param[inout] data   The data structure with all necessary CG vectors preallocated
  @param[inout] b      The known right hand side vector
  @param[inout] x      The solution vector to be computed in future CG iteration
  @param[inout] xexact The exact solution vector

  @return returns 0 upon success and non-zero otherwise

  @see GenerateGeometry
  @see GenerateProblem
*/
int OptimizeProblem(SparseMatrix & A, CGData & data, Vector & b, Vector & x, Vector & xexact) {

  // This function can be used to completely transform any part of the data structures.
  // Right now it does nothing, so compiling with a check for unused variables results in complaints

  SparseMatrix * curLevelMatrix = &A;
	while (curLevelMatrix != NULL) {
		OptimizeProblem_naive(*curLevelMatrix);
    curLevelMatrix = curLevelMatrix->Ac; // Make the just-constructed coarse grid the next level
  }


#if defined(HPCG_USE_MULTICOLORING)
  const local_int_t nrow = A.localNumberOfRows;
  std::vector<local_int_t> colors(nrow, nrow); // value `nrow' means `uninitialized'; initialized colors go from 0 to nrow-1
  int totalColors = 1;
  colors[0] = 0; // first point gets color 0

  // Finds colors in a greedy (a likely non-optimal) fashion.

  for (local_int_t i=1; i < nrow; ++i) {
    if (colors[i] == nrow) { // if color not assigned
      std::vector<int> assigned(totalColors, 0);
      int currentlyAssigned = 0;
      const local_int_t * const currentColIndices = A.mtxIndL[i];
      const int currentNumberOfNonzeros = A.nonzerosInRow[i];

      for (int j=0; j< currentNumberOfNonzeros; j++) { // scan neighbors
        local_int_t curCol = currentColIndices[j];
        if (curCol < i) { // if this point has an assigned color (points beyond `i' are unassigned)
          if (assigned[colors[curCol]] == 0)
            currentlyAssigned += 1;
          assigned[colors[curCol]] = 1; // this color has been used before by `curCol' point
        } // else // could take advantage of indices being sorted
      }

      if (currentlyAssigned < totalColors) { // if there is at least one color left to use
        for (int j=0; j < totalColors; ++j)  // try all current colors
          if (assigned[j] == 0) { // if no neighbor with this color
            colors[i] = j;
            break;
          }
      } else {
        if (colors[i] == nrow) {
          colors[i] = totalColors;
          totalColors += 1;
        }
      }
    }
  }

  std::vector<local_int_t> counters(totalColors);
  for (local_int_t i=0; i<nrow; ++i)
    counters[colors[i]]++;

  // form in-place prefix scan
  local_int_t old=counters[0], old0;
  for (local_int_t i=1; i < totalColors; ++i) {
    old0 = counters[i];
    counters[i] = counters[i-1] + old;
    old = old0;
  }
  counters[0] = 0;

  // translate `colors' into a permutation
  for (local_int_t i=0; i<nrow; ++i) // for each color `c'
    colors[i] = counters[colors[i]]++;
#endif

  return 0;
}

int OptimizeProblem_naive(SparseMatrix &A) {
  const local_int_t nrow = A.localNumberOfRows;
	int *j_before_diagonal = new int[nrow];
	int *j_after_diagonal = new int[nrow];

#ifndef HPCG_NO_OPENMP
	#pragma omp for
#endif
	for (local_int_t i=0; i < nrow; i++) {
    const local_int_t * const currentColIndices = A.mtxIndL[i];
    const int currentNumberOfNonzeros = A.nonzerosInRow[i];
		int last_j = -1;

    for (local_int_t j=0; j< currentNumberOfNonzeros; j++) {
			local_int_t jj = currentColIndices[j];
			if (jj==i) {
					j_before_diagonal[i] = last_j;
					j_after_diagonal[i] = j+1;
					break;
			}
			last_j = j;
		}
	}
	void * optimizationData = malloc(sizeof(void *)*2);
	((int **)optimizationData)[0] = j_before_diagonal;
	((int **)optimizationData)[1] = j_after_diagonal;
  A.optimizationData = optimizationData;

	return 0;
}

// Helper function (see OptimizeProblem.hpp for details)
double OptimizeProblemMemoryUse(const SparseMatrix & A) {

	double total = 0.0;
  const SparseMatrix * curLevelMatrix = &A;
	while (curLevelMatrix != NULL) {
  const local_int_t nrow = A.localNumberOfRows;
		double this_level = A.localNumberOfRows * (2*sizeof(int));
		total += this_level;
    curLevelMatrix = curLevelMatrix->Ac;
  }
  return total;

}
