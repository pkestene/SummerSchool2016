// linear algebra subroutines
// Ben Cumming @ CSCS

#ifndef LINALG_H
#define LINALG_H

#include "data.h"
#include "data_warehouse.h"

#include "kokkos_shared.h" // for DataArray1d, DataArray2d

namespace linalg
{

/**
 * Execution context for conjuguate Gradient.
 */
class CG_Solver {

public:
  CG_Solver(DataWarehouse &dw_);
  ~CG_Solver();

  // initialize temporary storage fields used by the cg solver
  // I do this here so that the fields are persistent between calls
  // to the CG solver. This is useful if we want to avoid malloc/free calls
  // on the device for the OpenACC implementation (feel free to suggest a better
  // method for doing this)
  void init(int nx, int ny);

  // conjugate gradient solver
  // solve the linear system A*x = b for x
  // the matrix A is implicit in the objective function for the diffusion equation
  // the value in x constitute the "first guess" at the solution
  // x(N)
  // ON ENTRY contains the initial guess for the solution
  // ON EXIT  contains the solution
  void ss_cg(Field2d x, Field2d const b, const int maxiters, const double tol, bool& success);

private:
  bool cg_initialized;
  DataWarehouse &dw;
  Field2d r, Ap, p, Fx, Fxold, v, xold; // 2d

  
}; // class CG_Solver

//////////////////////////////////////////////////////////////////////////
//  blas level 1 reductions
//////////////////////////////////////////////////////////////////////////

// computes the inner product of x and y
// x and y are vectors on length N
double ss_dot(Field2d x, Field2d y);

// computes the 2-norm of x
// x is a vector on length N
double ss_norm2(Field2d x);

// sets entries in a vector to value
// x is a vector on length N
// value is th
void ss_fill(Field1d x,     double value);
void ss_fill(Field2dHost x, double value);
void ss_fill(Field2d x,     double value);

//////////////////////////////////////////////////////////////////////////
//  blas level 1 vector-vector operations
//////////////////////////////////////////////////////////////////////////

// computes y := alpha*x + y
// x and y are vectors on length N
// alpha is a scalar
void ss_axpy(Field2d y, double alpha, Field2d x);

// computes y = x + alpha*(l-r)
// y, x, l and r are vectors of length N
// alpha is a scalar
void ss_add_scaled_diff(Field2d y, Field2d x, double alpha,
			Field2d l, Field2d r);

// computes y = alpha*(l-r)
// y, l and r are vectors of length N
// alpha is a scalar
void ss_scaled_diff(Field2d y, double alpha,
		    Field2d l, Field2d r);

// computes y := alpha*x
// alpha is scalar
// y and x are vectors on length n
void ss_scale(Field2d y, double alpha, Field2d x);

// computes linear combination of two vectors y := alpha*x + beta*z
// alpha and beta are scalar
// y, x and z are vectors on length n
void ss_lcomb(Field2d y, double alpha, Field2d x, double beta,
	      Field2d z);

// copy one vector into another y := x
// x and y are vectors of length N
void ss_copy(Field2d y, Field2d x);

} // namespace linalg

#endif // LINALG_H

