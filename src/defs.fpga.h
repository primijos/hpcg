#ifndef FPGA_H
#define FPGA_H
/*!
  This defines the type for integers that have local subdomain dimension.

  Define as "long long" when local problem dimension is > 2^31
*/
typedef int local_int_t;
//typedef long long local_int_t;

/*!
  This defines the type for integers that have global dimension

  Define as "long long" when global problem dimension is > 2^31
*/
#ifdef HPCG_NO_LONG_LONG
typedef int global_int_t;
#else
typedef long long global_int_t;
#endif

// This macro should be defined if the global_int_t is not long long
// in order to stop complaints from non-C++11 compliant compilers.
//#define HPCG_NO_LONG_LONG

// XXX TODO Check if this value can be dynamic at runtime (is here now
// because localmem needs to know it at compile time)
//#define BB 512
#define BB 8
static const unsigned int NNZ_PER_ROW=27;
static const unsigned int VECTOR_OPS_BLOCK=BB;
static const unsigned int WAXPBY_BLOCK=BB;
static const unsigned int DOTPRODUCT_BLOCK=BB;
static const unsigned int REST_BLOCK=BB;
static const unsigned int PROL_BLOCK=BB;
static const unsigned int SPMV_BLOCK=BB;
static const unsigned int SPMV_BLOCK_M=SPMV_BLOCK*NNZ_PER_ROW;
static const unsigned int SYMGS_BLOCK=BB;
static const unsigned int SYMGS_BLOCK_M=SYMGS_BLOCK*NNZ_PER_ROW;
#endif
