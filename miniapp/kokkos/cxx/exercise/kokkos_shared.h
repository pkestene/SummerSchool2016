#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

#ifdef CUDA
# define DEVICE Kokkos::Cuda
#include <cuda.h>
#endif

#ifdef OPENMP
# define DEVICE Kokkos::OpenMP
#endif

// define a default device
#ifndef DEVICE
# define DEVICE Kokkos::OpenMP
#endif

// Alias name for 1d and 2d array (Kokkos view)
typedef Kokkos::View<double*, DEVICE> Field1d;
typedef Field1d::HostMirror           Field1dHost;

typedef Kokkos::View<double**, DEVICE> Field2d;
typedef Field2d::HostMirror            Field2dHost;

//typedef Kokkos::View<double*, DEVICE> Field;
//typedef Field::HostMirror             FieldHost;

// Memory layout typedef
typedef typename Field2d::array_layout array_layout;

/**
 * Retrieve cartesian coordiante from index, using memory layout information.
 */
KOKKOS_INLINE_FUNCTION
void index2coord(int index, int &i, int &j, int Nx, int Ny) {
  if (std::is_same<array_layout,Kokkos::LayoutLeft>::value) { // Left <-> CUDA
    j = index / Nx;
    i = index - j*Nx;
  } else { // Right <-> OpenMP
    i = index / Ny;
    j = index - i*Ny;
  }
}


#endif // KOKKOS_SHARED_H_
