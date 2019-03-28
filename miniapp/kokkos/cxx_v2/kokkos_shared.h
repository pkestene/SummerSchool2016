#ifndef KOKKOS_SHARED_H_
#define KOKKOS_SHARED_H_

#include <Kokkos_Core.hpp>
#include <Kokkos_Parallel.hpp>
#include <Kokkos_View.hpp>

using Device = Kokkos::DefaultExecutionSpace;

// Alias name for 1d and 2d array (Kokkos view)
typedef Kokkos::View<double*, Device> Field1d;
typedef Field1d::HostMirror           Field1dHost;

// typedef Kokkos::View<double**, Device> Field2d;
// typedef Field2d::HostMirror            Field2dHost;

// typedef Kokkos::View<double*, Device> Field;
// typedef Field::HostMirror             FieldHost;


/**
 * Retrieve cartesian coordiante from index, using memory layout information.
 */
KOKKOS_INLINE_FUNCTION
void index2coord(int index, int &i, int &j, int Nx, int Ny) {
#ifdef KOKKOS_ENABLE_CUDA
    j = index / Nx;
    i = index - j*Nx;
#else
    i = index / Ny;
    j = index - i*Ny;
#endif
}


#endif // KOKKOS_SHARED_H_
