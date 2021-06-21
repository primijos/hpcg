/*!
 @file Vector.cpp

 HPCG data operations for dense vectors @ fpga
 */

#include <Vector.hpp>

#include "defs.fpga.h"
/*!
  Fill the input vector with zero values.

  @param[inout] v - On entrance v is initialized, on exit all its values are zero.
 */
#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_inout([localLength]vv)
#endif
#pragma omp task inout([localLength]vv)
void zero_vector_fpga(local_int_t localLength, double *vv) {
  for (int i=0; i<localLength; ++i) vv[i] = 0.0;
  return;
}

void ZeroVector_nw(Vector & v) {
  local_int_t localLength = v.localLength;
  double * vv = v.values;
	zero_vector_fpga(localLength,vv);
#pragma omp taskwait noflush
	return;
}

/*!
  Copy input vector to output vector.

  @param[in] v Input vector
  @param[in] w Output vector
 */
#ifndef OMPSS_ONLY_SMP
#pragma omp target device(fpga) num_instances(1) \
	 copy_inout([dstLocalLength]wv) copy_in([localLength]vv)
#endif
#pragma omp task in([localLength]vv) inout([dstLocalLength]wv)
void copy_vector_fpga(local_int_t localLength, local_int_t dstLocalLength, double *vv, double *wv) {
  for (int i=0; i<localLength; ++i) wv[i] = vv[i];
}

void CopyVector_nw(const Vector & v, Vector & w) {
  local_int_t localLength = v.localLength;
  assert(w.localLength >= localLength);
  double * vv = v.values;
  double * wv = w.values;

	copy_vector_fpga(localLength,w.localLength,vv,wv);
#pragma omp taskwait noflush
	return;
}
