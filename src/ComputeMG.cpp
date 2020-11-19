
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
	 copy_in([REST_BLOCK]f2c) copy_inout([REST_BLOCK]rcv) localmem_copies
#endif
#pragma omp task inout([REST_BLOCK]rcv) in([rfl]rfv,[REST_BLOCK]f2c,[Axfl]Axfv) no_copy_deps
void compute_restriction_fpga_block(double *rcv, double *rfv, local_int_t *f2c, double *Axfv, local_int_t Axfl, local_int_t rfl) {
  for (local_int_t i=0; i<REST_BLOCK; ++i) rcv[i] = rfv[f2c[i]] - Axfv[f2c[i]];
}

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([rfl]rfv,[nc]f2c,[Axfl]Axfv) copy_inout([nc]rcv)
#endif
#pragma omp task inout([nc]rcv) in([rfl]rfv,[nc]f2c,[Axfl]Axfv)
void compute_restriction_fpga(local_int_t nc, double *rcv, double *rfv, local_int_t *f2c, double *Axfv, local_int_t Axfl, local_int_t rfl) {
	local_int_t nblocks = nc / REST_BLOCK;
	// XXX TODO check for block sizes non-divisible by n
	int remainder = nc % REST_BLOCK;

	for (local_int_t i=0;i<nblocks;i++) {
		double *_rcv = rcv + i*REST_BLOCK;
		local_int_t *_f2c = f2c + i*REST_BLOCK;
		compute_restriction_fpga_block(_rcv,rfv,_f2c,Axfv,Axfl,rfl);
	}
}

int ComputeRestriction(const SparseMatrix & A, const Vector & rf) {

  double * Axfv = A.mgData->Axf->values;
  double * rfv = rf.values;
  double * rcv = A.mgData->rc->values;
  local_int_t * f2c = A.mgData->f2cOperator;
  local_int_t nc = A.mgData->rc->localLength;

	local_int_t rfl = rf.localLength;
  local_int_t Axfl = A.mgData->Axf->localLength;
	// XXX TODO check for block sizes non-divisible by n
	assert(nc % REST_BLOCK == 0);

	compute_restriction_fpga(nc,rcv,rfv,f2c,Axfv,Axfl,rfl);
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
	 copy_in([PROL_BLOCK]xcv,[PROL_BLOCK]f2c) copy_inout([xfl]xfv) localmem_copies
#endif
#pragma omp task inout([xfl]xfv) in([PROL_BLOCK]xcv,[PROL_BLOCK]f2c) no_copy_deps
void compute_prolongation_fpga_block(double *xfv, double *xcv, local_int_t *f2c, local_int_t xfl) {
  for (local_int_t i=0; i<PROL_BLOCK; ++i) xfv[f2c[i]] += xcv[i]; // This loop is safe to vectorize
}

#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_in([nc]xcv,[nc]f2c) copy_inout([xfl]xfv)
#endif
#pragma omp task inout([xfl]xfv) in([nc]xcv,[nc]f2c)
void compute_prolongation_fpga(local_int_t nc, double *xfv, double *xcv, local_int_t *f2c, local_int_t xfl) {
	local_int_t nblocks = nc / PROL_BLOCK;
	// XXX TODO check for block sizes non-divisible by n
	int remainder = nc % PROL_BLOCK;

	for (local_int_t i=0;i<nblocks;i++) {
		double *_xcv = xcv + i*PROL_BLOCK;
		local_int_t *_f2c = f2c + i*PROL_BLOCK;
		compute_prolongation_fpga_block(xfv, _xcv, _f2c, xfl);
	}
}

int ComputeProlongation(const SparseMatrix & Af, Vector & xf) {

  double * xfv = xf.values;
  double * xcv = Af.mgData->xc->values;
  local_int_t * f2c = Af.mgData->f2cOperator;
  local_int_t nc = Af.mgData->rc->localLength;

  local_int_t xfl = xf.localLength;
	// XXX TODO check for block sizes non-divisible by n
	assert(nc % PROL_BLOCK == 0);

	compute_prolongation_fpga(nc,xfv,xcv,f2c,xfl);
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
int ComputeMG(const SparseMatrix  & A, const Vector & r, Vector & x) {

  assert(x.localLength==A.localNumberOfColumns); // Make sure x contain space for halo values

  ZeroVector(x); // initialize x to zero

  int ierr = 0;
  if (A.mgData!=0) { // Go to next coarse level if defined
    int numberOfPresmootherSteps = A.mgData->numberOfPresmootherSteps;
    for (int i=0; i< numberOfPresmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
    ierr = ComputeSPMV(A, x, *A.mgData->Axf); if (ierr!=0) return ierr;
    // Perform restriction operation using simple injection
    ierr = ComputeRestriction(A, r);  if (ierr!=0) return ierr;
    ierr = ComputeMG(*A.Ac,*A.mgData->rc, *A.mgData->xc);  if (ierr!=0) return ierr;
    ierr = ComputeProlongation(A, x);  if (ierr!=0) return ierr;
    int numberOfPostsmootherSteps = A.mgData->numberOfPostsmootherSteps;
    for (int i=0; i< numberOfPostsmootherSteps; ++i) ierr += ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
  }
  else {
    ierr = ComputeSYMGS(A, r, x);
    if (ierr!=0) return ierr;
  }
  return 0;
}
