//******************************************
// operators.f90
// based on min-app code written by Oliver Fuhrer, MeteoSwiss
// modified by Ben Cumming, CSCS
// *****************************************

// Description: Contains simple operators which can be used on 3d-meshes

#include "data.h"
#include "operators.h"
#include "stats.h"

namespace operators {

void diffusion(Field1d U, Field1d S, const DataWarehouse &dw)
{

  using data::options;
  
  const Field1d &bndE = dw.bndE;
  const Field1d &bndW = dw.bndW;
  const Field1d &bndN = dw.bndN;
  const Field1d &bndS = dw.bndS;
  
  const Field1d& x_old = dw.x_old;
  
  double dxs = 1000. * (options.dx * options.dx);
  double alpha = options.alpha;
  int nx = options.nx;
  int ny = options.ny;
  int iend  = nx - 1;
  int jend  = ny - 1;
    
  // the interior grid points
  Kokkos::parallel_for( nx*ny, KOKKOS_LAMBDA( int index ) {
      int i,j;
      j = index / nx;
      i = index - j*nx;
      //index2coord(index, i, j, nx, ny);

      if (i>0    && j>0 &&
	  i<iend && j<jend) {
	S(index) = -(4. + alpha) * U(index)               // central point
	  + U(index-1)  + U(index+1) // east and west
	  + U(index-nx) + U(index+nx) // north and south
	  + alpha * x_old(index)
	  + dxs * U(index) * (1.0 - U(index));
      }
    });
    
  // the east boundary
  {
    int i = nx - 1;

    Kokkos::parallel_for( ny-2, KOKKOS_LAMBDA( int k ) {
	int j = k + 1;
	int index = i + nx*j;
	{
	  S(index) = -(4. + alpha) * U(index)
	    + U(index-1) + U(index-nx) + U(index+nx)
	    + alpha * x_old(index) + bndE(j)
	    + dxs * U(index) * (1.0 - U(index));
	}
      });
  }
    
  // the west boundary
  {
    int i = 0;

    Kokkos::parallel_for( ny-2, KOKKOS_LAMBDA( int k ) {
	int j = k + 1;
	int index = i + nx*j;
	{
	  S(index) = -(4. + alpha) * U(index)
	    + U(i+1,j) + U(i,j-1) + U(i,j+1)
	    + alpha * x_old(index) + bndW(j)
	    + dxs * U(index) * (1.0 - U(index));
	}
      });
  }
    
  // the north boundary (plus NE and NW corners)
  {
    int j = ny - 1;
            
    // north boundary
    Kokkos::parallel_for( nx, KOKKOS_LAMBDA( int i ) {
	int index = i + nx*j;
	if (i == 0) {
	  // NW corner
	  S(index) = -(4. + alpha) * U(index)
	    + U(index+1) + U(index-nx)
	    + alpha * x_old(index) + bndW(j) + bndN(i)
	    + dxs * U(index) * (1.0 - U(index));
	} else if (i == nx-1) {
	  // NE corner
	  S(index) = -(4. + alpha) * U(index)
	    + U(index-1) + U(index-nx)
	    + alpha * x_old(index) + bndE(j) + bndN(i)
	    + dxs * U(index) * (1.0 - U(index));
	} else {
	  S(index) = -(4. + alpha) * U(index)
	    + U(index-1) + U(index+1) + U(index-nx)
	    + alpha * x_old(index) + bndN(i)
	    + dxs * U(index) * (1.0 - U(index));
	}
      
      });
  }
  
  // the south boundary
  {
    int j = 0;
            
    // south boundary
    //for (int i = 1; i < iend; i++)
    Kokkos::parallel_for( nx, KOKKOS_LAMBDA( int i ) {
	int index = i + nx*j;
	if (i == 0) {
	  // SW corner
	  S(index) = -(4. + alpha) * U(index)
	    + U(index+1) + U(index+nx)
	    + alpha * x_old(index) + bndW(j) + bndS(i)
	    + dxs * U(index) * (1.0 - U(index));
	} else if (i == nx-1) {
	  // SE corner
	  S(index) = -(4. + alpha) * U(index)
	    + U(index-1) + U(index+nx)
	    + alpha * x_old(index) + bndE(j) + bndS(i)
	    + dxs * U(index) * (1.0 - U(index));
	} else {
	  S(index) = -(4. + alpha) * U(index)
	    + U(index-1) + U(index+1) + U(index+nx)
	    + alpha * x_old(index) + bndS(i)
	    + dxs * U(index) * (1.0 - U(index));
	}
	
      });
  }
    
  // Accumulate the flop counts
  // 8 ops total per point
  stats::flops_diff +=
    + 12 * (nx - 2) * (ny - 2) // interior points
    + 11 * (nx - 2  +  ny - 2) // NESW boundary points
    + 11 * 4;                  // corner points
}

} // namespace operators
