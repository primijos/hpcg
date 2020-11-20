
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
 @file ComputeMG.cpp

 HPCG routine
 */

#include "ComputeMG.hpp"
#include "ComputeSYMGS.hpp"
#include "ComputeSPMV.hpp"



/*!
#include "ComputeRestriction_ref.hpp"
  Routine to compute the coarse residual vector.

  @param[inout]  A - Sparse matrix object containing pointers to mgData->Axf, the fine grid matrix-vector product and mgData->rc the coarse residual vector.
  @param[in]    rf - Fine grid RHS.


  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
#include "defs.fpga.h"
#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([rfl]rfv,[nc]f2c,[Axfl]Axfv) copy_inout([nc]rcv)
#endif
#pragma omp task inout([nc]rcv) in([rfl]rfv,[nc]f2c,[Axfl]Axfv)
void compute_restriction_fpga(local_int_t nc, double *rcv, double *rfv, local_int_t *f2c, double *Axfv, local_int_t Axfl, local_int_t rfl) {
  for (local_int_t i=0; i<nc; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
}

int ComputeRestriction_nw(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

	local_int_t rfl = rf.localLength;
  local_int_t Axfl = A.mgData->Axf->localLength;

	compute_restriction_fpga(nc,rcv,rfv,f2c,Axfv,Axfl,rfl);

  return 0;
}
int ComputeRestriction(const SparseMatrix & A, const Vector & rf) {
	ComputeRestriction_nw(A,rf);
#pragma omp taskwait
	return 0;
}

/*!
#include "ComputeProlongation_ref.hpp"
  Routine to compute the coarse residual vector.

  @param[in]  Af - Fine grid sparse matrix object containing pointers to current coarse grid correction and the f2c operator.
  @param[inout] xf - Fine grid solution vector, update with coarse grid correction.

  Note that the fine grid residual is never explicitly constructed.
  We only compute it for the fine grid points that will be injected into corresponding coarse grid points.

  @return Returns zero on success and a non-zero value otherwise.
*/
#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([nc]xcv,[nc]f2c) copy_inout([xfl]xfv)
#endif
#pragma omp task inout([xfl]xfv) in([nc]xcv,[nc]f2c)
void compute_prolongation_fpga(local_int_t nc, double *xfv, double *xcv, local_int_t *f2c, local_int_t xfl) {
  for (local_int_t i=0; i<nc; ++i) xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize
}
int ComputeProlongation_nw(const SparseMatrix & Af, Vector & xf) {

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

  local_int_t xfl = xf.localLength;

	compute_prolongation_fpga(nc,xfv,xcv,f2c,xfl);

  return 0;
}
int ComputeProlongation(const SparseMatrix & Af, Vector & xf) {
	ComputeProlongation_nw(Af,xf);
#pragma omp taskwait
	return 0;
}

/*!
  @param[in] A the known system matrix
  @param[in] r the input vector
  @param[inout] x On exit contains the result of the multigrid V-cycle with r as the RHS, x is the approximation to Ax = r.

  @return returns 0 upon success and non-zero otherwise

  @see ComputeMG_ref
*/
int ComputeMG_nw(const SparseMatrix  & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector_nw(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS_nw(A, r, x);
    if (ierr!=0) return ierr;
    ierr = ComputeSPMV_nw(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction_nw(A, r);  if (ierr!=0) return ierr;
    ierr = ComputeMG_nw(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation_nw(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS_nw(A, r, x);
    if (ierr!=0) return ierr;
  }
  else {
    ierr = ComputeSYMGS_nw(A, r, x);
    if (ierr!=0) return ierr;
  }
#pragma omp taskwait noflush
  return 0;
}


int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {
	ComputeMG_nw(A,r,x);
#pragma omp taskwait
  return 0;
}
